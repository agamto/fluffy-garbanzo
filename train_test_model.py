from lstm_model import VanillaLSTM
import torch.nn as nn
import torch
def train_model(device,data_train,lables,modelType='v-lstm',epoches=5,lost_function=nn.MSELoss,optimizer='adam',learning_rate=0.01,path_to_save='./'):

    if modelType == 'v-lstm':
        model = VanillaLSTM(len(data_train[0]),len(data_train)).to(device)
    lossfunc = lost_function.to(device)
    optimizer_func = torch.optim.Adam(model.parameters(),lr=learning_rate)
    if optimizer == 'sgd':
        optimizer_func = torch.optim.SGD(model.parameters(),lr=learning_rate)

    for epoch in range(epoches):
        print('EPOCH {}:'.format(epoch + 1))
        #using the model
        tried = model(data_train)
        #calculation of loss function
        loss = lossfunc(tried, lables)
        #backpropogation
        print('LOSS :{}'.format(loss))
        optimizer_func.zero_grad()
        loss.backward()
        optimizer_func.step()
    model.eval()
    torch.save(model,path_to_save)
def test_model(model_path,data_test,device):
    model = torch.load(model_path)
    test_correct = 0
    total = 0
    with torch.no_grad():
        for data, lables in data_test:
            outputs = model(data.to(device))
            _, predicted = torch.max(outputs, 1)
            total += lables.size(0)
            test_correct += (predicted == lables).sum().item()
    print(f'accurecy ={100 * test_correct / total}%')
