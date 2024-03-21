import yaml
import os
import torch
import evaluate

bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

def load_model(model_name, classes=None):
    pass

def get_scheduler(config, optimizer):

    if config['scheduler'] == 'ReduceOnPlateu':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    elif config['scheduler'] == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    elif config['scheduler'] == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config['eta_min'])
    
    elif config['scheduler'] == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])
    
    elif config['scheduler'] == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = config['T_0'], T_mult=config['T_m'], eta_min=config['eta_min'])
    
    else:
        raise Exception("Scheduler not found")

def get_optimer(optimizer_name, model, lr):
    if optimizer_name == 'ADAM':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'ADAGRAD':
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise Exception("Optimizer not found")

def createDir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def LoadConfig(test_name):
    with open("/fhome/gia07/Image_captioning/src/setups/" + test_name + ".yaml") as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt[test_name] = test_name
    opt["output_dir"] = "/fhome/gia07/Image_captioning/src/runs/" + test_name + "/images/"
    opt["train_output_dir"] = "/fhome/gia07/Image_captioning/src/runs/" + test_name + "/train_images/"
    opt["weights_dir"] = "/fhome/gia07/Image_captioning/src/runs/" + test_name + "/weights/"
    opt["root_dir"] = "/fhome/gia07/Image_captioning/src/runs/" + test_name + "/"

    createDir(opt["root_dir"])
    createDir(opt["weights_dir"])
    createDir(opt["output_dir"])
    createDir(opt["train_output_dir"])

    # aixo es per evitar que falli amb configs antics
    opt["network"]["params"]["teacher_forcing_ratio"] = opt["network"].get("params", {}).get("teacher_forcing_ratio", opt.get("teacher_forcing_ratio", 0))
    opt["network"]["params"]["rnn_layers"] = opt["network"].get("params", {}).get("rnn_layers",3)

    # opt["network"]["checkpoint"] = opt["network"].get("checkpoint", None)

    return opt

def get_weights(weights_path):
    path2load = None
    for path in os.listdir(weights_path):
        if path[-4:] == ".pth":
            if path2load == None: 
                path2load = path
            elif int(path.split(".")[0]) > int(path2load.split(".")[0]):        
                path2load = path
    if path2load == None: 
        print("No weights found")
        return None

    print("Loading weights: ", path2load)
    return weights_path + path2load

def metrics_evaluation(pred, ref):
    metrics = {}
    try:
        metrics["bleu1"] = bleu.compute(predictions=pred, references=ref, max_order=1)["bleu"]
        metrics["bleu2"] = bleu.compute(predictions=pred, references=ref, max_order=2)["bleu"]
        metrics["rougeL"] = rouge.compute(predictions=pred, references=ref)["rougeL"]
        metrics["meteor"] = meteor.compute(predictions=pred, references=ref)["meteor"]
    except ZeroDivisionError:
        metrics["bleu1"] = 0
        metrics["bleu2"] = 0
        metrics["rougeL"] = 0
        metrics["meteor"] = 0
    return metrics
    