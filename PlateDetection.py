#coding=utf-8
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import sys
import cv2
import StringIO
import base64

from flask import Flask, abort, request, jsonify
app = Flask(__name__) # 测试数据暂时存放
tasks = []
label = ['Z', 'G', 'E', '2', '\xe9\xb2\x81', '5', '3', '\xe9\x99\x95', '6', 'C', 'F', '\xe8\xb1\xab', 'R', '0',
             'X', '4', 'J', 'K', 'A', 'N', 'W', 'P', '7', '1', '\xe4\xba\xac', 'V', 'D', 'L', 'Q', 'S', 'M', '8',
             '\xe7\xb2\xa4', 'B', '9', 'Y', 'T', 'H', 'U']

def recognize(label, imgs):
    labelnum = len(label)

    result = []
    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # 定义第一个卷积层
    W_conv1 = tf.Variable(tf.truncated_normal([7, 7, 1, 32], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    L1_conv = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    L1_relu = tf.nn.relu(L1_conv + b_conv1)
    L1_pool = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 定义第二个卷积层
    W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    L2_conv = tf.nn.conv2d(L1_pool, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    L2_relu = tf.nn.relu(L2_conv + b_conv2)
    L2_pool = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 全连接层
    W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    h_pool2_flat = tf.reshape(L2_pool, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # readout层
    W_fc2 = tf.Variable(tf.truncated_normal([1024, labelnum], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[labelnum]))
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    output = tf.argmax(y_conv, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './model/model.ckpt')
        for i in range(len(imgs)):
            img = cv2.resize(imgs[i], (28, 28), cv2.INTER_LINEAR);
            img = np.array(img).reshape(-1);
            input = [img]
            index = sess.run(output, feed_dict={x: input, keep_prob: 1.0})[0]
            result.append(label[index])
    return result


def detect(orgimg):
    orgimg = base64.b64decode(orgimg)
    orgimg = Image.open(StringIO.StringIO(orgimg))
    #orgimg = orgimg.transpose(Image.ROTATE_180)
    orgimg = cv2.cvtColor(np.asarray(orgimg), cv2.COLOR_RGB2BGR)
    h = orgimg.shape[0]
    w = orgimg.shape[1]
    orgimg = cv2.resize(orgimg, (400 * w // h, 400), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('org', orgimg)
    hsv = cv2.cvtColor(orgimg, cv2.COLOR_BGR2HSV) #转到HSV空间
    lower = np.array([100, 100, 50])
    upper = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower, upper) #通过蓝色抠出车牌区域
    kernel = np.ones((5, 19), np.uint8)
    openingimg = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('bw', openingimg)
    contours = cv2.findContours(openingimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fail = True
    for i in range(len(contours[1])):
        contour = contours[1][i]
        rect = cv2.minAreaRect(contour)
        if ((rect[1][0] < 10) | (rect[1][1] < 10)): #根据长宽比和角度筛选出车牌区域
            continue
        if ((rect[1][0] / rect[1][1] > 7) | (rect[1][0] / rect[1][1] < 2.5)) & (
                (rect[1][1] / rect[1][0] > 7) | (rect[1][1] / rect[1][0] < 2.5)):
            continue
        if (rect[2] > -15):
            rx = rect[0][0]
            ry = rect[0][1]
            M = cv2.getRotationMatrix2D((rx, ry), rect[2], 1.0)
            orgimg = cv2.warpAffine(orgimg, M, (400 * w // h, 400))
            x1 = rect[0][1] - rect[1][1] // 2
            x2 = rect[0][1] + rect[1][1] // 2
            y1 = rect[0][0] - rect[1][0] // 2
            y2 = rect[0][0] + rect[1][0] // 2
            result = orgimg[int(x1):int(x2), int(y1):int(y2)]
            fail = False
            break
        if (rect[2] < -75):
            rx = rect[0][0]
            ry = rect[0][1]
            M = cv2.getRotationMatrix2D((rx, ry), 90 + rect[2], 1.0)
            orgimg = cv2.warpAffine(orgimg, M, (400 * w // h, 400))
            x1 = rect[0][0] - rect[1][1] // 2
            x2 = rect[0][0] + rect[1][1] // 2
            y1 = rect[0][1] - rect[1][0] // 2
            y2 = rect[0][1] + rect[1][0] // 2
            result = orgimg[int(y1):int(y2), int(x1):int(x2)]
            fail = False
            break

    if fail:
        print file, 'fail'
        return None
    #cv2.imshow('r', result)
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('plant', img_gray)
    if img_gray is None:
        return None
    img_thre = img_gray.copy()
    ret, img_thre = cv2.threshold(img_thre, 0, 255, cv2.THRESH_OTSU)
    # 分割字符
    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = img_thre.shape[0]
    width = img_thre.shape[1]
    white_max = 0
    black_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if img_thre[j][i] == 255:
                s += 1
            if img_thre[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)

    def find_end(start_):
        end_ = start_ + 1
        for m in range(start_ + 1, width):
            if (black[m] > 10 * white[m]):
                end_ = m
                break
            if (m == width - 1):
                end_ = m
        return end_

    def fill(img):
        w = img.shape[1]
        h = img.shape[0]
        if w > h:
            return False, img
        result = np.zeros((h, h))
        result[0:, (h - w) // 2:(h - w) // 2 + w] = img[:, :]
        return True, result

    n = 1
    start = 1
    end = 2
    starts = []
    ends = []
    while n < width - 2:
        n += 1
        start = n
        end = find_end(start)
        n = end
        if end - start > 4:
            starts.append(start)
            ends.append(end)
    divide = []
    divide.append(starts[0])
    for i in range(len(starts) - 1):
        divide.append((ends[i] + starts[i + 1]) // 2)
    divide.append(ends[len(ends)-1])
    results = []
    distance = []
    while divide[1]-divide[0]<10:
        divide.remove(divide[1])
    for i in range(len(divide) - 1):
        distance.append(divide[i+1]-divide[i])
    while len(distance)>7:
        ind = distance.index(min(distance))
        if ind==0:
            divide.remove(divide[1])
        elif ind==len(distance)-1 :
            divide.remove(divide[ind])
        elif distance[ind+1]>distance[ind-1]:
            divide.remove(divide[ind])
        else:
            divide.remove(divide[ind+1])
        for i in range(len(divide) - 1):
            distance=[]
            distance.append(divide[i + 1] - divide[i])
    if(len(distance)==6):
        ind = distance.index(max(distance))
        divide.insert(ind+1,(divide[ind+1]+divide[ind])//2)
    for i in range(len(divide) - 1):
        img = img_thre[1:height, divide[i]:divide[i + 1]]
        flag, img = fill(img)
        if img is None:
            print 'error'
            break
        (h, w) = img.shape
        wnum = 0
        for a in range(h):
            for b in range(w):
                if img[a, b] == 255:
                    wnum += 1
        if (float(wnum) / float(w * h)) < 0.05:
            continue
        #cv2.imwrite(str(i), img)
        results.append(img)
    return results

@app.route('/add_task', methods=['POST'])
def add_task():
    if not request.json or 'licensePlate' not in request.json: abort(400)
    task = { 'licensePlate': request.json['licensePlate']}
    imgs = detect(request.json['licensePlate'])
    s = recognize(label, imgs)
    r = ''
    if len(s) > 2:
        for i in range(1, len(s)):
            r += s[i]
    tasks.append(task)
    return jsonify(s[0]+r)

@app.route('/get_task/', methods=['GET'])
def get_task():
    if not request.args or 'id' not in request.args:
        return jsonify(tasks)
    else:
        task_id = request.args['id']
        task = filter(lambda t: t['id'] == int(task_id), tasks)
        return jsonify(task) if task else jsonify({'result': 'not found'})

if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(host="0.0.0.0", port=8383, debug=True)