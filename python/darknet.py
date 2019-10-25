from ctypes import *
import math
import random
import cv2
import time

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("/content/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/mnt/DATA_1/1_GitHub/darknet-master/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    # im = load_image(image, 0, 0)
    im = nparray_to_image(image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)

    return image

def GetColour(className):
    if className == "Car":
        color = (0,255,0) #Green
    elif className == "Truck":
        color = (255,0,0) #Red
    elif className == "Motorbike":
        color = (255,255,0) #unknow.
    else:
        color = (0,0,255) #Blue
    return color



# Define destination resolution
Destination_W = 416
Destination_H = 416
    
if __name__ == "__main__":
    # net = load_net("/content/darknet/cfg/yolov3-tiny.cfg", "/content/darknet/backup/yolov3-tiny_final.weights", 0)
    net = load_net(b"cfg/yolov3-tiny.cfg", b"backup/KHTN_yolov3-tiny_final.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    # net = load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
    # meta = load_meta("/content/darknet/cfg/VehiclesDetection.data")
    meta = load_meta(b"cfg/VehiclesDetection.data")
    # meta = load_meta("/content/darknet/cfg/ChienTestYolo.data")
    cap_temp = cv2.VideoCapture('/mnt/DATA_1/Data/VehiclesDetection/LayData_Lan_1/20190922_135448.MOV')
    
    frameCouter = 0
    writer = None
    TimeCountFPS = None
    CountFPS = 0
    
    while True:
        if TimeCountFPS is None:
            TimeCountFPS = time.time()
            CountFPS = 0

        (grabbed, frame) = cap_temp.read()
        #frame = cv2.resize(frame, (Destination_W,Destination_H), interpolation = cv2.INTER_AREA)
		
        # cv2.imwrite("image.jpg", frame)
        # r = detect(net, meta, "image.jpg")
        r = detect(net, meta, frame)
        frameCouter += 1
        for j in range(0,len(r)):
            # print r[j]
            className = r[j][0].decode('utf-8')
            prob = r[j][1]           
            
            x = r[j][2][0]
            y = r[j][2][1]
            w = r[j][2][2]
            h = r[j][2][3]
            x_min = int(x - (w/2))
            y_min = int(y - (h/2))
            x_max = int(x + (w/2))
            y_max = int(y + (h/2))
            
            # Draw
            # color = (255,106,77) #Orange
            # print(className)
            color = GetColour(className)
            cv2.putText(frame, className, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            # cv2.imwrite("python/output/Cam0_frame-{}.png".format(frameCouter), frame)


            cv2.putText(frame, "chien.dotruong@gmail.com test", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,0,0), 3)

        if writer is None:
            # initialize our video writer
            # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            fourcc = cv2.VideoWriter_fourcc(*"MPEG")
            # writer = cv2.VideoWriter('python/output.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
            # writer = cv2.VideoWriter('python/output.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
            writer = cv2.VideoWriter('python/output.mp4', fourcc, 60, (frame.shape[1], frame.shape[0]), True)
            
        writer.write(frame)
        # print("Frame: " + str(frameCouter))

        CountFPS += 1
        if time.time() - TimeCountFPS >= 1:
            print('--------------- FPS: %d' % (CountFPS))
            TimeCountFPS = None

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
cap.release()