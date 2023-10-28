from ultralytics import YOLO
import os
import time

currentDir = os.getcwd()
print(currentDir)
# result loop
model = YOLO("best.pt")
# get the path/directory
print("input sample name:")
inputsample = str(input())
folder_dir = currentDir+"/"+inputsample+"/"
pictureDir = currentDir+"/"+inputsample+"/frames_rgb/"

# Check whether the specified path exists or not
dir1Exists = os.path.exists(folder_dir)

if not dir1Exists:
   os.makedirs(folder_dir)
dir2Exists = os.path.exists(folder_dir + "/frames_output/")

if not dir2Exists:
   os.makedirs(folder_dir + "/frames_output/")

output = []
track = []
maxBack = 20
sizeCap = 0.15
sortedImages = sorted(os.listdir(pictureDir))
imageCount = len(os.listdir(pictureDir))
totalCounters = {"wood": 0, "glas": 0, "plastic": 0, "metal": 0}
start = time.process_time()
for cIma in range(imageCount):
    image = sortedImages[cIma]
    singleInfo = []
    # check if the image ends with png
    if image.endswith(".png"):
        print(image)
        counters = {"wood": 0, "glas": 0, "plastic": 0, "metal": 0}
        results = model.predict(pictureDir + image, conf=0.2, max_det=30)
        for i in range(len(results[0].boxes)):
            box = results[0].boxes[i]
            boxInfo = {"label": "", "size": "", "pos": "", "probs": ""}
            label = results[0].names[box.cls[0].item()]
            boxInfo["label"] = label
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            boxInfo["size"] = [x2 - x1, y2 - y1]
            boxInfo["pos"] = [x_center, y_center]
            boxInfo["probs"] = results[0].boxes[i].conf.detach().numpy()[0]
            if label == "wood" and boxInfo["probs"] > 0.73 \
                    or label == "glas" and boxInfo["probs"] > 0.76 \
                    or label == "plastic" and boxInfo["probs"] > 0.48 \
                    or label == "metal" and boxInfo["probs"] > 0.35:
                singleInfo.append(boxInfo)
                counters[label] += 1
                found = False
                for t in range(max(0, cIma - maxBack), cIma):
                    if t < cIma:
                        infoBlock = output[t]
                        for inB in range(len(infoBlock)):
                            labelBack = infoBlock[inB]["label"]
                            labelNow = boxInfo["label"]
                            size1Back = infoBlock[inB]["size"][0]
                            size1Now = boxInfo["size"][0]
                            size1Med = size1Back + size1Now/2
                            size2Back = infoBlock[inB]["size"][1]
                            size2Now = boxInfo["size"][1]
                            size2Med = size1Back + size1Now/2
                            yBack = infoBlock[inB]["pos"][1]
                            yNow = boxInfo["pos"][1]
                            xBack = infoBlock[inB]["pos"][0]
                            xNow = boxInfo["pos"][0]
                            if labelBack == labelNow and abs(size1Back - size1Now) < size1Med*sizeCap \
                                    and abs(size2Back - size2Now) < size2Med*sizeCap \
                                    and abs(yBack - yNow) < 5 \
                                    and abs(xNow - (xBack + 27 * (cIma - t))) < 5 * (cIma - t):
                                found = True
                                break
                        if found:
                            break
                if not found:
                    track.append(boxInfo)
                    totalCounters[label] += 1

        with open(folder_dir + "/frames_output/" + image[:-3] + 'txt', 'w') as f:
            for key in counters:
                f.write(str(counters[key]))
                f.write("\n")
    output.append(singleInfo)
timeCalculation = time.process_time() - start
time.sleep(0.2)
with open(folder_dir + "/output.txt", 'w') as f:
    for key in totalCounters:
        f.write(str(totalCounters[key]))
        f.write("\n")
with open(folder_dir + "/time.txt", 'w') as f:
    f.write(str(timeCalculation))

