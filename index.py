import detector, utils, PIL.Image as pil, torchvision.transforms as transforms, time
device = "cuda"
print(device)
detectionModel = utils.load_model(detector.Detector(),"index.pt", device=device)
detectionModel.eval()
detectionModel = detectionModel.to(device)
image = pil.open("frame0000.jpg") 

image = transforms.ToTensor()(image) # shape: (3,720,1280)
image = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(image)
image = image.unsqueeze(0)
image = image.to(device)
start = time.time()
for i in range(N:=10):
    inference = detectionModel(image).cpu()
print(N, "inferences took ", time.time()-start," seconds. This is a frequency of ", N/(time.time()-start), " Hz")
