import copy
import torch
import logging
from datasets.data_loading import get_test_loader
from conf import get_num_classes  # Ensure this is properly defined in 'conf'
from utils.registry import ADAPTATION_REGISTRY
import wandb

wandb.init(project="federated_tta", name="FederatedTTA")
import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import wandb

import methods
from conf import cfg, ckpt_path_to_domain_seq, get_num_classes, load_cfg_from_args
from datasets.data_loading import get_test_loader
from models.model import get_model
from utils.eval_utils import eval_domain_dict, get_accuracy
from utils.misc import print_memory_info
from utils.registry import ADAPTATION_REGISTRY

logger = logging.getLogger(__name__)


def federated_avg(models):
    """Performs federated averaging on a list of models."""
    global_model = copy.deepcopy(models[0])
    for key in global_model.state_dict():
        global_model.state_dict()[key] = torch.mean(
            torch.stack([model.state_dict()[key].float() for model in models]), dim=0
        )
    return global_model

class Client:
    def __init__(self, model, setting, domain_sequence, severities, cfg, model_preprocess, device):
        # Setup test-time adaptation method
        available_adaptations = ADAPTATION_REGISTRY.registered_names()
        assert cfg.MODEL.ADAPTATION in available_adaptations, \
            f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
        self.model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=model, num_classes=get_num_classes(cfg.CORRUPTION.DATASET))
        print(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")

        self.setting = setting
        self.domain_sequence = domain_sequence
        self.severities = severities
        self.cfg = cfg
        self.model_preprocess = model_preprocess
        self.device = device
        self.current_domain_idx = 0
        self.current_step = 0
        self.accuracies = []
        self.dataloader = self._get_dataloader()
        self.dataloader_iter = iter(self.dataloader)  # Initialize iterator

    def _get_dataloader(self):
        if self.setting == "mixed":
            domain_name = "mixed"
            self.setting = "mixed_domains"
        else:
            domain_name = self.domain_sequence[self.current_domain_idx]
        severity = self.severities[0]
       
        print(self.setting)
        # print(domain_name)
        # print(self.domain_sequence)
        # print("severity ", severity)
        # print(self.cfg.CORRUPTION.DATASET)


        #domain_name = self.domain_sequence[self.current_domain_idx]
        
        return get_test_loader(
            setting=self.setting,
            adaptation=self.cfg.MODEL.ADAPTATION,
            dataset_name=self.cfg.CORRUPTION.DATASET,
            preprocess=self.model_preprocess,
            data_root_dir=self.cfg.DATA_DIR,
            domain_name=domain_name,
            domain_names_all=self.domain_sequence,
            severity=severity,
            num_examples=self.cfg.CORRUPTION.NUM_EX,
            rng_seed=self.cfg.RNG_SEED,
            use_clip=self.cfg.MODEL.USE_CLIP,
            n_views=self.cfg.TEST.N_AUGMENTATIONS,
            delta_dirichlet=self.cfg.TEST.DELTA_DIRICHLET,
            batch_size=self.cfg.TEST.BATCH_SIZE,
            shuffle=False,
            workers=min(self.cfg.TEST.NUM_WORKERS, os.cpu_count())
        )

    def process_batch(self):
        """Processes a single batch of data."""
        self.current_step = self.current_step + 1
        self.model.to(self.device)
        self.model.eval()  # Ensure the model is in evaluation mode
        #print(self.current_domain_idx)
        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            
            if self.setting in ["continual", "reset_each_shift"]:
                if self.setting == "reset_each_shift":
                    print("*"*100)
                    try:
                        self.model.reset()
                        logger.info("resetting model")
                    except AttributeError:
                        logger.warning("not resetting model")

              
                self.current_domain_idx = (self.current_domain_idx + 1) % len(self.domain_sequence)
                self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)

        imgs, labels = batch[0].to(self.device), batch[1].long().to(self.device)  # Convert labels to Long dtype
        outputs = self.model(imgs)
        batch_accuracy = (outputs.argmax(1) == labels).float().mean().item()
        #print(batch_accuracy)
        self.accuracies.append(batch_accuracy)
        
        if self.current_step % 30 == 0:
            print(self.current_step)
            print("Client {id(self)}")
            print("actual")
            print(labels)
            print("predicted")
            print(outputs.argmax(1))

        wandb.log({f"Client {id(self)} {self.setting} Batch Accuracy": batch_accuracy})

        return self.model.state_dict(), batch_accuracy

    def overall_accuracy(self):
        """Calculates the overall accuracy after all steps."""
        overall_acc = sum(self.accuracies) / len(self.accuracies) if self.accuracies else 0.0
        wandb.log({f"Client {id(self)} Overall Accuracy": overall_acc})
        return overall_acc


def run_federated_tta(clients, cfg):
    """Run federated test-time adaptation for 750 steps."""
    global_model = copy.deepcopy(clients[0].model)

    for step in range(750):
        print(f"Federated step {step+1}/750")
        client_models = []
        batch_accuracies = []

        for client in clients:
            model_state, batch_accuracy = client.process_batch()
            client_models.append(copy.deepcopy(client.model))
            batch_accuracies.append(batch_accuracy)

        if cfg.fed.fed_tech == "fedavg":
            # Perform federated averaging
            global_model = federated_avg(client_models)

            # Update clients with the global model
            for client in clients:
                client.model.load_state_dict(global_model.state_dict())

        avg_batch_accuracy = sum(batch_accuracies) / len(batch_accuracies)
        logger.info(f"Step {step+1}: Average Batch Accuracy: {avg_batch_accuracy:.2%}")
        wandb.log({"Global Average Batch Accuracy": avg_batch_accuracy})

    # Calculate and log overall accuracy for each client
    total_avg = []
    for i, client in enumerate(clients):
        overall_acc = client.overall_accuracy()
        total_avg.append(overall_acc)
        logger.info(f"Client {i+1} Overall Accuracy: {overall_acc:.2%}")

    # Calculate the total average
    if total_avg:
        avg_accuracy = sum(total_avg) / len(total_avg)
        logger.info(f"Total Average Accuracy: {avg_accuracy:.2%}")
    else:
        logger.info("No clients available to calculate the total average.")


if __name__ == "__main__":
    load_cfg_from_args('fed tta')
    # User-defined settings
    n = 10  # Total clients
    n1, n2, n3, n4 = 0, 0 , 0, 10   # Clients for continual, mixed, reset_each_shift
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # Initialize base model and preprocessing
    base_model, model_preprocess = get_model(cfg, num_classes, device)

    # Initialize models and clients
    models = [copy.deepcopy(base_model) for _ in range(n)]
    domain_sequence = cfg.CORRUPTION.TYPE
    severities = cfg.CORRUPTION.SEVERITY
    

    #cfg.MODEL.ADAPTATION = 'source'
    
    clients = [
        Client(model, setting, domain_sequence, severities, cfg, model_preprocess, device)
        for model, setting in zip(
            models,
            ["continual"] * n1 + ["mixed"] * n2 + ["reset_each_shift"] * n3 + ["correlated"] * n4
        )
    ]

    print(cfg.TEST.BATCH_SIZE)
    
    
   
     # setup wandb logging
     
    wandb.run.name = "fed-" + cfg.MODEL.ADAPTATION + "-" + cfg.fed.fed_tech + "-" + cfg.CORRUPTION.DATASET + "-dirchlet " +  str(cfg.TEST.DELTA_DIRICHLET)

    information = "10 correlated"
    wandb.run.name += "-" + information

    # add current bangladesh time to the run name
    now = datetime.now()
    new_time = now + timedelta(hours=11)
    wandb.run.name += "-" + new_time.strftime("%Y-%m-%d-%H-%M-%S")

    wandb.config.update(cfg)
    

    # Run federated test-time adaptation
    run_federated_tta(clients, cfg)
