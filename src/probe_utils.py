from utils import load_npy
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import load_train_data, load_test_data, sync_shuffle

def convert_score2acc(score,label):
    binary_pred = (score >= 0.5).astype(int)
    accuracy = np.mean(binary_pred == label)
    return accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCA_Probe:
    def __init__(self, n_components=1, whiten=True):
        self.n_components = n_components
        self.whiten = whiten
        self.directions = None
        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self.lr = None
        self.scaler = StandardScaler()
    def fit(self, data, label):
        data = self.scaler.fit_transform(data)
        self.pca.fit(data)
        direction = self.pca.components_.squeeze()
        temp = self.pca.transform(data)
        self.lr = LogisticRegression(solver='liblinear')
        self.lr.fit(temp, label)
        coeff = np.sign(self.lr.coef_).squeeze()
        self.directions = coeff * direction
    def pred(self, data):
        temp = self.pca.transform(data)
        return self.lr.predict(temp)
    
    def get_direction(self, layer):
        return self.directions[layer]
    
class MMProbe(torch.nn.Module):
    def __init__(self, direction, covariance=None, inv=None, atol=1e-3):
        super().__init__()
        self.direction = torch.nn.Parameter(direction, requires_grad=False)
    def forward(self, x, iid=False):
        if isinstance(x, np.ndarray):
            x =  torch.from_numpy(x)
        x = x.to(device)
        if iid:
            return torch.nn.Sigmoid()(x @ self.inv @ self.direction)
        else:
            return torch.nn.Sigmoid()(x @ self.direction)

    def pred(self, x, iid=False):
        return self(x, iid=iid).round()

    def from_data(acts, labels, atol=1e-3, device='cpu'):
        if isinstance(acts, np.ndarray):
            acts = torch.from_numpy(acts)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
            labels = labels.float()
        acts, labels = acts.to(device), labels.to(device)
        pos_acts, neg_acts = acts[labels==1], acts[labels==0]

        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean
        
        centered_data = torch.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
        covariance = centered_data.t() @ centered_data / acts.shape[0]
        
        probe = MMProbe(direction, covariance=covariance).to(device)

        return probe
class MLP_Probe(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, iid=None):
        if isinstance(x, np.ndarray):
            x =  torch.from_numpy(x)
        x = x.to(device)
        return self.net(x).squeeze(-1)

    def pred(self, x, iid=None):
        return self(x).round()
    
    def from_data(acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        if isinstance(acts, np.ndarray):
            acts = torch.from_numpy(acts)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
            labels = labels.float()
        acts, labels = acts.to(device), labels.to(device)
        probe = MLP_Probe(acts.shape[-1]).to(device)
        
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for i in range(epochs):
            opt.zero_grad()
            loss = torch.nn.BCELoss()(probe(acts), labels)
            loss.backward()
            opt.step()
            if (i+1) % 400 == 0 or i == 0 or i+1 == 1000:
                print(f"Epoch [{i+1}/{epochs}], Loss: {loss.item()}") 
        return probe


def test_probe(probe, test_file, posi, model_name, index, features_folder):
    name = model_name+'_'+test_file+'_'+posi+'.npy'
    data = load_npy(file_name=name, features_folder=features_folder)
    label = load_npy(file_name=model_name+'_'+test_file+'_labels.npy', features_folder=features_folder)
    if 'head' in posi:
        bsz = data.shape[0]
        dim = data.shape[3]
        num_layers = data.shape[1]
        num_heads = data.shape[2]
        data =  data.reshape(bsz, num_layers*num_heads, dim)
    test_data = data[:,index,:]
    
    y_test_pred = probe.predict(test_data)
    acc = accuracy_score(y_test_pred, label)
    return acc

def test_probe_data(probe, test_data, label):
    if not isinstance(probe, list):
        probe = [probe]
    acc_list = []
    if len(test_data.shape) == 4:
        bsz, _, _, head_dim = test_data.shape
        test_data = test_data.reshape(bsz, -1, head_dim)
    if len(test_data.shape) == 3:
        for i in  range(len(probe)):
            p = probe[i]
            temp_test = test_data[:,i,:]
            if isinstance(p, MLP_Probe) or isinstance(p, MMProbe):
                pred = p(temp_test).cpu().detach().numpy()
                acc = convert_score2acc(pred, label)
                
            else:
                pred = p.predict(temp_test)
                acc = accuracy_score(pred, label)
            acc_list.append(acc)
        return np.array(acc_list)
    elif len(test_data.shape) ==2:
        assert len(probe)==1, "wrong setting, only one format data, but have multiple probes"
        p = probe[0]
        if isinstance(p, MLP_Probe):
            pred = p(test_data).cpu().detach().numpy()
            acc = convert_score2acc(pred, label)
        elif isinstance(p, MMProbe):
            pred = p(test_data).cpu().detach().numpy()
            acc = convert_score2acc(pred, label)
        else:
            pred = p.predict(test_data)
            acc = accuracy_score(pred, label)
        return acc
    else:
        assert False, f"wrong test_data shape 2 or 3 is ok but  {len(test_data.shape)}"
        
def train_head_probes(seed, all_X_train, all_X_val, y_train, y_val, num_layers=-1, num_heads=-1,solver=None, penalty=None, is_MLP=False, posi_list=None, is_MM=False, is_PCA=False):
    """
    input: all_X_train:  bsz x layers x num_head x dim
    """
    all_head_accs = []
    probes = []
    if num_layers == -1:
        num_layers = all_X_train.shape[1]
    if num_heads == -1:
        num_heads = all_X_train.shape[2]
    if posi_list!=None and not isinstance(posi_list, np.ndarray):
        print(f"posi_list len: {len(posi_list)} --- shape: {len(posi_list[0].shape)}")
        posi_list =  np.array(posi_list)
        assert posi_list.shape[0]==num_heads*num_layers, f"Wrong posi list num {posi_list.shape[0]} however need {num_heads*num_layers}"
        print(f"training with dim {posi_list.shape[1]}")
    current_index = 0
    first_output=True
    for layer in tqdm(range(num_layers)): 
        for head in range(num_heads): 
            if posi_list is None:
                X_train = all_X_train[:,layer,head,:]
                X_val = all_X_val[:,layer,head,:]
            else:
                X_train = all_X_train[:,layer,head,posi_list[current_index]]
                X_val = all_X_val[:,layer,head,posi_list[current_index]]
                current_index+=1
                
            if is_MLP == False and is_MM == False and is_PCA == False:
                if first_output:
                    print("-----------training LR---------------")
                if solver!=None:
                    if penalty!=None:
                        clf = LogisticRegression(random_state=seed, max_iter=10000, solver=solver,penalty=penalty).fit(X_train, y_train)
                    else:
                        clf = LogisticRegression(random_state=seed, max_iter=10000, solver=solver).fit(X_train, y_train)
                else:
                    clf = LogisticRegression(random_state=seed, max_iter=10000, solver='lbfgs').fit(X_train, y_train)
                y_pred = clf.predict(X_train)
                y_val_pred = clf.predict(X_val)
                # print(y_val_pred)
                # print(clf.predict_proba(X_val))
                # return
                all_head_accs.append(accuracy_score(y_val, y_val_pred))
                probes.append(clf)
            elif is_MLP == True:
                if first_output:
                    print("----------training MLP-----------------")
                probe = MLP_Probe.from_data(X_train, y_train, device=device)
                pred = probe(X_val)
                if torch.is_tensor(pred):
                    pred = pred.cpu().detach().numpy()
                acc = convert_score2acc(pred, y_val)
                all_head_accs.append(acc)
                probes.append(probe)
            elif is_MM == True:
                if first_output:
                    print("-----------training MM------------")
                probe = MMProbe.from_data(X_train, y_train, device=device)
                pred = probe(X_val)
                if torch.is_tensor(pred):
                    pred = pred.cpu().detach().numpy()
                acc = convert_score2acc(pred, y_val)
                all_head_accs.append(acc)
                probes.append(probe)
            elif is_PCA == True:
                if first_output:
                    print("-----------training PCA---------------")
                probe = PCA_Probe(n_components=1, whiten=True)
                probe.fit(X_train, y_train)         
                pred = probe.pred(X_val)
                if torch.is_tensor(pred):
                    pred = pred.cpu().detach().numpy()
                all_head_accs.append(accuracy_score(y_val, pred))
                probes.append(probe)
            first_output=False
    all_head_accs_np = np.array(all_head_accs)

    return probes, all_head_accs_np

def train_probes(seed, all_X_train, all_X_val, y_train, y_val, num_layers=-1,solver=None, penalty=None, is_MLP=False, posi_list=None, is_MM=False, is_PCA=False):
    """
    separated_activation:  num x layers x dim
    """
    all_head_accs = []
    probes = []
    if num_layers == -1:
        num_layers = all_X_train.shape[1]
    train_num = all_X_train.shape[0]
    vali_num = all_X_val.shape[0]
    print(f"training.....  num  {train_num}")
    print(f"testing.... num {vali_num}  ")
    assert all_X_train.shape[0] == y_train.shape[0], "wrong number match of trian data and label"
    assert all_X_val.shape[0] == y_val.shape[0], "wrong number match of val data and label"
    
    if posi_list!=None and not isinstance(posi_list, np.ndarray):
        # print(posi_list)
        print(f"posi_list len: {len(posi_list)} --- shape: {len(posi_list[0].shape)}")
        posi_list =  np.array(posi_list)
        assert posi_list.shape[0]==num_layers, f"Wrong posi list num {posi_list.shape[0]} however need {num_layers}"
        print(f"training with dim {posi_list.shape[1]}")
    
    first_output=True
    for layer in tqdm(range(num_layers)): 
        if posi_list is None:
            X_train = all_X_train[:,layer,:]
            X_val = all_X_val[:,layer,:]
        else:
            X_train = all_X_train[:,layer,posi_list[layer]]
            X_val = all_X_val[:,layer,posi_list[layer]] 
                       
        if is_MLP == False and is_MM == False and is_PCA == False:
            if first_output:
                print("-----------training LR---------------")
            if solver!=None:
                if penalty!=None:
                    clf = LogisticRegression(random_state=seed, max_iter=10000, solver=solver,penalty=penalty).fit(X_train, y_train)
                else:
                    clf = LogisticRegression(random_state=seed, max_iter=10000, solver=solver).fit(X_train, y_train)
            else:
                clf = LogisticRegression(random_state=seed, max_iter=10000, solver='lbfgs').fit(X_train, y_train)
            y_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)
            
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            probes.append(clf)
        elif is_MLP == True:
            if first_output:
                print("-----------training MLP---------------")
            probe = MLP_Probe.from_data(X_train, y_train, device=device)           
            pred = probe(X_val)
            if torch.is_tensor(pred):
                pred = pred.cpu().detach().numpy()
            acc = convert_score2acc(pred, y_val)
            all_head_accs.append(acc)
            probes.append(probe)
        elif is_MM == True:
            if first_output:
                print("-----------training MM---------------")
            probe = MMProbe.from_data(X_train, y_train, device=device)           
            pred = probe(X_val)
            if torch.is_tensor(pred):
                pred = pred.cpu().detach().numpy()
            acc = convert_score2acc(pred, y_val)
            all_head_accs.append(acc)
            probes.append(probe)
        elif is_PCA == True:
            if first_output:
                print("-----------training PCA---------------")
            probe = PCA_Probe(n_components=1, whiten=True)
            probe.fit(X_train, y_train)         
            pred = probe.pred(X_val)
            if torch.is_tensor(pred):
                pred = pred.cpu().detach().numpy()
            all_head_accs.append(accuracy_score(y_val, pred))
            probes.append(probe)
        first_output = False
    all_head_accs_np = np.array(all_head_accs)

    return probes, all_head_accs_np


def train_single_probe( x_train, y_train, x_val=None, y_val=None, solver=None, penalty=None, seed=0, is_MLP=False, is_MM=False, is_PCA=False):
    if is_MLP == False and is_MM == False and is_PCA == False:
        print("-----------training single LR--------------")
        if solver!=None:
            if penalty!=None:
                clf = LogisticRegression(random_state=seed, max_iter=10000, solver=solver,penalty=penalty).fit(x_train, y_train)
            else:
                clf = LogisticRegression(random_state=seed, max_iter=10000, solver=solver).fit(x_train, y_train)
        else:
            clf = LogisticRegression(random_state=seed, max_iter=10000, solver='lbfgs').fit(x_train, y_train)
        y_pred = clf.predict(x_train)
        if x_val is not None:
            y_val_pred = clf.predict(x_val)
            # print(y_val_pred)
            # print(clf.predict_proba(X_val))
            # return
            acc = accuracy_score(y_val, y_val_pred)
            return clf, acc
        else:
            return clf
    elif is_MLP == True:
        print("---------training single MLP--------------")
        probe = MLP_Probe.from_data(x_train, y_train, device=device)
        if x_val is not None:
            pred = probe(x_val)
            if torch.is_tensor(pred):
                pred = pred.cpu().detach().numpy()
            acc = convert_score2acc(pred, y_val)
            return probe, acc
        else:
            return probe
    elif is_MM == True:
        print("----------training single MM------------")
        probe = MMProbe.from_data(x_train, y_train, device=device)
        if x_val is not None:
            pred = probe(x_val)
            if torch.is_tensor(pred):
                pred = pred.cpu().detach().numpy()
            acc = convert_score2acc(pred, y_val)
            return probe, acc
        else:
            return probe
    elif is_PCA == True:
        print("-----------training PCA---------------")
        probe = PCA_Probe(n_components=1)
        probe.fit(x_train, y_train)   
        if x_val is not None:      
            pred = probe.pred(x_val)
            if torch.is_tensor(pred):
                pred = pred.cpu().detach().numpy()
            acc = accuracy_score(y_val, pred)
            return probe, acc
        else:
            return probe
        
def get_probe_acc(model_name, dataset_name, posi='mlp_wise',num_heads=32,  test_file=None, solver=None, penalty=None, is_MLP=False, posi_list=None, merge_test=False, is_MM=False
                  , selected_vali=False, selected_test=False, is_PCA=False, train_upperbound=-1):
    """_summary_

    Args:
        model_name (_type_): _description_
        dataset_name (_type_): _description_
        posi (str, optional): _description_. Defaults to 'mlp_wise'.
        num_heads (int, optional): _description_. Defaults to 32.
        portion (float, optional): _description_. Defaults to 0.7.
        test_file (_type_, optional): _description_. Defaults to None.
        solver (_type_, optional): _description_. Defaults to None.
        penalty (_type_, optional): _description_. Defaults to None.
        is_MLP (bool, optional): _description_. Defaults to False.
        posi_list (_type_, optional): _description_. Defaults to None.
        merge_test (bool, optional): _description_. Defaults to False.
        is_MM (bool, optional): _description_. Defaults to False.
        selected_vali (bool, optional): _description_. Defaults to False.
        selected_test (bool, optional): _description_. Defaults to False.
        is_PCA (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if model_name == 'llama2' or model_name == 'llama2_7b' or model_name == 'llama2_7b_chat':
        num_heads=32
    elif model_name == 'llama2_13b' or model_name == 'llama2_13b_chat':
        num_heads = 40
    if isinstance(dataset_name,str):
        train_data, train_label = load_train_data(model_name=model_name, dataset_name=dataset_name, posi=posi, 
             num_heads=num_heads,upper_bound=train_upperbound )
    else:
        assert isinstance(dataset_name,list), "Wrong dataset name defination"
        train_data_list = []
        train_label_list = []
        for temp_dataset_name in dataset_name:
            temp_data, temp_label = load_train_data(model_name=model_name, dataset_name=temp_dataset_name, posi=posi,
             num_heads=num_heads,upper_bound=train_upperbound )
            train_data_list.append(temp_data)
            train_label_list.append(temp_label)
        train_data = np.concatenate(train_data_list, axis=0)
        train_label = np.concatenate(train_label_list, axis=0)
        train_data, train_label = sync_shuffle(train_data, train_label)

    if merge_test == True:
        #-- merge all vali data------------
        vali_data_list = []
        vali_label_list = []
        if test_file!=None and isinstance(test_file, str):
            test_file = [test_file]
        
        for other_file in test_file:
            vali_data, vali_label = load_test_data(model_name=model_name, test_file=other_file, posi=posi, num_heads=num_heads, selected_vali=selected_vali, selected_test=selected_test)
            vali_data_list.append(vali_data)
            vali_label_list.append(vali_label)
        #  ---- merge done ----
        vali_data = np.concatenate(vali_data_list, axis=0)
        vali_label = np.concatenate(vali_label_list, axis=0)
        vali_data, vali_label = sync_shuffle(vali_data, vali_label)
        # there we get the vali acc
        if posi == 'head_wise':
            probs, acc_list = train_head_probes(seed=0, all_X_train=train_data, all_X_val=vali_data, y_train=train_label, y_val=vali_label, num_layers=-1,solver=solver, penalty=penalty, is_MLP=is_MLP, posi_list=posi_list, is_MM=is_MM, is_PCA=is_PCA)
        else:
            probs, acc_list = train_probes(seed=0, all_X_train=train_data, all_X_val=vali_data, y_train=train_label, y_val=vali_label, num_layers=-1,solver=solver, penalty=penalty, is_MLP=is_MLP, posi_list=posi_list, is_MM=is_MM, is_PCA=is_PCA)
        return probs, acc_list
    else:
        ans_probs = []
        ans_acc = []
        is_First = True

        if test_file!=None and isinstance(test_file, str):
            test_file = [test_file]
        if test_file!=None:
            for other_file in test_file:
                vali_data, vali_label = load_test_data(model_name=model_name, test_file=other_file, posi=posi, num_heads=num_heads, selected_vali=selected_vali, selected_test=selected_test)
                if is_First:
                    if posi == 'head_wise':
                        probs, acc_list = train_head_probes(seed=0, all_X_train=train_data, all_X_val=vali_data, y_train=train_label, y_val=vali_label, num_layers=-1,solver=solver, penalty=penalty, is_MLP=is_MLP, posi_list=posi_list, is_MM=is_MM, is_PCA=is_PCA)
                    else:
                        probs, acc_list = train_probes(seed=0, all_X_train=train_data, all_X_val=vali_data, y_train=train_label, y_val=vali_label, num_layers=-1,solver=solver, penalty=penalty, is_MLP=is_MLP, posi_list=posi_list, is_MM=is_MM, is_PCA=is_PCA)
                    ans_probs.append(probs)
                    ans_acc.append(acc_list)
                    print("train done!")
                    is_First = False
                    del train_data
                    del vali_data
                else:
                    acc_list = test_probe_data(probs, vali_data, vali_label)
                    ans_acc.append(acc_list)
                    
        if len(ans_acc)==1:
            return ans_probs[0], ans_acc[0]
        else:
            return ans_probs, ans_acc
