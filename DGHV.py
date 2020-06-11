import cv2
import random
import numpy as np
from collections import Counter

def str2bits(m):
    data = ''
    for i in m:
        data += format(ord(i), 'b').zfill(8)
    return data

def bits2str(m):
    data = ''
    for i in range(0, len(m), 8):
        data += chr(int(m[i:i+8], 2))
    return data

def key_gen(num):
    pub_key = []
    for i in range(num+1):
        q = random.randint((1<<30), (1<<35))
        r = random.randint(1, (1<<9))
        pub_key.append(p*q+(1<<9)*r)
    argmax = max(range(len(pub_key)), key=pub_key.__getitem__)
    pub_key[0], pub_key[argmax] = pub_key[argmax], pub_key[0]
    return pub_key

def E(m, pub_key):
    r = random.randint(1, (1<<9))
    s = random.choice(pub_key[1:])
    return (s+(1<<9)*r+m)%pub_key[0]

def enc(img, pub_key):
    h, w = img.shape
    enc_img = [ [0 for j in range(w)] for i in range(h)]
    for i in range(h):
        for j in range(w//2):
            r = random.randint(1, (1<<9))
            s = sum(random.sample(pub_key[1:], 3))
            enc_img[i][2*j] = (s+(1<<9)*r+image[i][2*j])%pub_key[0]
            enc_img[i][2*j+1] = (s+(1<<9)*r+image[i][2*j+1])%pub_key[0]
    return enc_img

def emb(img, bitstr, pub_key):
    h = len(img)
    w = len(img[0])
    emb_img = [ [0 for j in range(w)] for i in range(h)]
    pos_diff = {}
    neg_diff = {}
    max_embed = 0
    for i in range(h):
        for j in range(w//2):
            diff = img[i][2*j]-img[i][2*j+1]
            if diff>0:
                if diff in pos_diff.keys():
                    pos_diff[diff] += 1
                else:
                    pos_diff[diff] = 1
            if diff<0:
                if diff in neg_diff.keys():
                    neg_diff[diff] += 1
                else:
                    neg_diff[diff] = 1
            
    counter = Counter(pos_diff).most_common(1)
    ec_pos = counter[0][0]
    max_embed += counter[0][1]
    counter = Counter(neg_diff).most_common(1)
    ec_neg = counter[0][0]
    max_embed += counter[0][1]
    embed_len = len(bitstr)
    if embed_len > max_embed:
        exit('embedded data too long')
    cnt = 0
    for i in range(h):
        for j in range(w//2):
            diff = img[i][2*j]-img[i][2*j+1]
            if diff >= 0:
                if diff > ec_pos:
                    emb_img[i][2*j] = (img[i][2*j]+E(1, pub_key))%pub_key[0]
                if diff < ec_pos:
                    emb_img[i][2*j] = (img[i][2*j]+E(0, pub_key))%pub_key[0]
                if diff == ec_pos:
                    if cnt < embed_len:
                        emb_img[i][2*j] = (img[i][2*j]+E(int(bitstr[cnt]), pub_key))%pub_key[0]
                        cnt += 1
                    else:
                        emb_img[i][2*j] = (img[i][2*j]+E(0, pub_key))%pub_key[0]
                emb_img[i][2*j+1] = (img[i][2*j+1]+E(0, pub_key))%pub_key[0]
            else:
                if diff < ec_neg:
                    emb_img[i][2*j+1] = (img[i][2*j+1]+E(1, pub_key))%pub_key[0]
                if diff > ec_neg:
                    emb_img[i][2*j+1] = (img[i][2*j+1]+E(0, pub_key))%pub_key[0]
                if diff == ec_neg:
                    if cnt < embed_len:
                        emb_img[i][2*j+1] = (img[i][2*j+1]+E(int(bitstr[cnt]), pub_key))%pub_key[0]
                        cnt += 1
                    else:
                        emb_img[i][2*j+1] = (img[i][2*j+1]+E(0, pub_key))%pub_key[0]
                emb_img[i][2*j] = (img[i][2*j]+E(0, pub_key))%pub_key[0]

    return emb_img, ec_pos, ec_neg, embed_len

def dec(img, p):
    h = len(img)
    w = len(img[0])
    dec_img = [ [0 for j in range(w)] for i in range(h)]
    for i in range(h):
        for j in range(w//2):
            dec_img[i][2*j] = (img[i][2*j]%p)%(1<<9)
            dec_img[i][2*j+1] = (img[i][2*j+1]%p)%(1<<9)
    return dec_img

def ext(img, pos, neg, length):
    bitstr = ''
    cnt = 0
    h = len(img)
    w = len(img[0])
    ext_img = [ [img[i][j] for j in range(w)] for i in range(h)]
    for i in range(h):
        for j in range(w//2):
            diff = img[i][2*j]-img[i][2*j+1]
            if diff >= 0:
                if diff >= pos+1:
                    ext_img[i][2*j] -= 1
                    if diff == pos+1 and cnt<length:
                        bitstr += '1'
                        cnt += 1
                else:
                    #ext_img[i][2*j] -= 0
                    if diff == pos and cnt<length:
                        bitstr += '0'
                        cnt += 1
                #ext_img[i][2*j+1] -= 0
            else:
                if diff <= neg-1:
                    ext_img[i][2*j+1] -= 1
                    if diff == neg-1 and cnt<length:
                        bitstr += '1'
                        cnt += 1
                else:
                    #ext_img[i][2*j+1] -= 0
                    if diff == neg and cnt<length:
                        bitstr += '0'
                        cnt += 1
                #ext_img[i][2*j] -= 0

    return ext_img, bitstr

if __name__ == '__main__':
    image = cv2.imread('lena.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('lena.png', image)
    image = np.array(image, dtype='int64')
    #print(image)

    p = random.randint((1<<20), (1<<22))*2+1
    pub_key = key_gen(10)

    image = enc(image, pub_key)
    img = np.zeros((len(image), len(image[0])))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = image[i][j]%256
    img = np.array(img, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('lena_encrypt.png', img)

    data = str2bits('data')
    image, pos, neg, length = emb(image, data, pub_key)
    #print(pos, neg, length)
    img = np.zeros((len(image), len(image[0])))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = image[i][j]%256
    img = np.array(img, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('lena_embedded.png', img)

    image = dec(image, p)
    img = np.array(image, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('lena_decrypt.png', img)
    
    image, bit_str = ext(image, pos, neg, length)
    img = np.array(image, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('lena_restore.png', img)
    print(bits2str(bit_str))
