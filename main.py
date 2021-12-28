import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torch.nn as nn
import torch.optim as optim
import Model
import time
import torch.optim.lr_scheduler as lr_scheduler
import cv2
from facedetector import FaceDetector
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'
trans = transforms.Compose([transforms.ToTensor()])

def train():
    learning_rate = 1e-4
    train_data = dset.ImageFolder(root="pictures/train_dataset", transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    data_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=8)
    net = Model.vgg19().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss().to(device)
    total_batch = len(data_loader)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=1/2)

    epochs = 1000
    for epoch in range(epochs):
        start = time.time()
        avg_cost = 0.0
        for num, data in enumerate(data_loader):
            batchstart = time.time()
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = net(imgs)
            loss = loss_func(out, labels)
            loss.backward() #backpropagation
            optimizer.step() #Weights update
            avg_cost += loss / total_batch
            print("epoch %d batch %d / 200 time : %.4f s" %(epoch + 1, num + 1, (time.time()-batchstart)))
        with open("loss.txt", 'a') as ps:
            ps.write(str(avg_cost) + "\n")
        scheduler.step()
        print(time.time()-start)
        print('[Epoch:{}] cost = {}'.format(epoch + 1, avg_cost))
        if(epoch % 10 == 0 and epoch != 0 ):
            torch.save(net.state_dict(), "./Checkpoint/checkpoint.pth")


    print('Learning Finished!')
    torch.save(net.state_dict(), "./Checkpoint/checkpoint.pth")



def test():

    inputs = [cv2.imread(file) for file in glob.glob("pictures/call/*.jpg")]
    labelColor = [(10, 0, 255), (10, 255, 0)]
    font = cv2.FONT_HERSHEY_COMPLEX
    for i in range(len(inputs)):
        img = inputs[i]
        facelist = []

        faceDetector = FaceDetector(
            prototype='models/deploy.prototxt.txt',
            model='models/res10_300x300_ssd_iter_140000.caffemodel',
        )
        faces = faceDetector.detect(img)

        croped_image = 0
        for face in faces:
            xStart, yStart, width, height = face
            facelist.append(face)
            xStart, yStart = max(xStart, 0), max(yStart, 0)
            faceImg = img[yStart:yStart + height, xStart:xStart + width]
            img = cv2.rectangle(img,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (255, 255, 255),
                          thickness=2)
            six_four_img = cv2.resize(faceImg, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('pictures/detection_croped/croped/%04d.png' % croped_image, six_four_img)
            croped_image = croped_image + 1

        trans = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_data = torchvision.datasets.ImageFolder(root='pictures/detection_croped/', transform=trans)
        test_set = DataLoader(dataset=test_data, batch_size=len(test_data))

        new_net = Model.vgg19().to(device)
        new_net.load_state_dict(torch.load('./Checkpoint/checkpoint.pth'))
        with torch.no_grad():
            for num, data in enumerate(test_set):
                imgs, label = data
                imgs = imgs.to(device)
                templist = []
                prediction = new_net(imgs)
                for k in range(len(test_data)):
                    if(torch.argmax(prediction[k]) == 1):
                        templist.append("mask off")
                    else:
                        templist.append("mask on")

        ## modle prediction

        for j in range(len(faces)):
            textSize = cv2.getTextSize(templist[j], font, 1, 2)[0]
            textX = facelist[j][0] + facelist[j][2] // 2 - textSize[0] // 2
            if templist[j] == "mask off":
                key = 0
                print("마스크 안 쓴 사람 감지")
            else:
                key = 1
            cv2.putText(img,
                        templist[j],
                        (textX, facelist[j][1] - 20),
                        font, 1, labelColor[key], 2)

        cv2.imshow('sadf', img)
        cv2.waitKey()
        cv2.imwrite('pictures/results/result%02d.png ' %i, img)

def test2():

    inputs = cv2.VideoCapture('1233.mp4')
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    outputs = cv2.VideoWriter('output.mp4', codec, inputs.get(5), (int(inputs.get(3)), int(inputs.get(4))))
    labelColor = [(10, 0, 255), (10, 255, 0)]
    font = cv2.FONT_HERSHEY_COMPLEX
    while True:
        ret, frame = inputs.read()
        facelist = []

        faceDetector = FaceDetector(
            prototype='models/deploy.prototxt.txt',
            model='models/res10_300x300_ssd_iter_140000.caffemodel',
        )
        faces = faceDetector.detect(frame)

        croped_image = 0
        for face in faces:
            xStart, yStart, width, height = face
            facelist.append(face)
            xStart, yStart = max(xStart, 0), max(yStart, 0)
            faceImg = frame[yStart:yStart + height, xStart:xStart + width]
            frame = cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (255, 255, 255),
                          thickness=2)
            six_four_img = cv2.resize(faceImg, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('pictures/detection_croped/croped/%04d.png' % croped_image, six_four_img)
            croped_image = croped_image + 1

        trans = torchvision.transforms.Compose([
            transforms.ToTensor(),])
        test_data = torchvision.datasets.ImageFolder(root='pictures/detection_croped/', transform=trans)
        test_set = DataLoader(dataset=test_data, batch_size=len(test_data))

        new_net = Model.vgg19().to(device)
        new_net.load_state_dict(torch.load('./Checkpoint/checkpoint.pth'))
        with torch.no_grad():
            for num, data in enumerate(test_set):
                imgs, label = data
                imgs = imgs.to(device)
                templist = []
                prediction = new_net(imgs)
                for k in range(len(test_data)):
                    if(torch.argmax(prediction[k]) == 1):
                        templist.append("mask off")
                    else:
                        templist.append("mask on")


        for j in range(len(faces)):
            textSize = cv2.getTextSize(templist[j], font, 1, 2)[0]
            textX = facelist[j][0] + facelist[j][2] // 2 - textSize[0] // 2
            if templist[j] == "mask off":
                key = 0
            else:
                key = 1

            cv2.putText(frame,
                        templist[j],
                        (textX, facelist[j][1] - 20),
                        font, 1, labelColor[key], 2)
        outputs.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(30) > 0:
            break
    inputs.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, test_pictures, test_video')
    args = parser.parse_args()

    if  args.mode == 'train':
        train()
    elif args.mode == 'test_pictures':
        test()
    elif args.mode == 'test_video':
        test2()
    else:
        raise Exception("Unknow --mode")

