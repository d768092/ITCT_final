import cv2
import random
import math
import Crypto.Util.number
import numpy as np
from collections import Counter

def mod_pow(a, x, n):
    ans = 1
    while(x>0):
        if x%2==1:
            ans = (ans*a)%n
        a = (a*a)%n
        x = x//2
    return ans

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

def key_gen(p, q):
    N = p*q
    lamb = (p-1)*(q-1)//math.gcd(p-1, q-1)
    while True:
        g = random.randint(1, N*N)
        if math.gcd(g, N)!=1:
            continue
        h = (mod_pow(g, lamb, N*N)-1)//N
        if math.gcd(h, N)==1:
            mu = Crypto.Util.number.inverse(h, N)
            break
    pri_key = (lamb, mu)
    pub_key = (N, g)
    return pri_key, pub_key

def E(m, pub_key, r=None):
    N, g = pub_key
    if r==None:
        while True:
            r = random.randint(1, N*N)
            if math.gcd(r, N)==1:
                break
    return (mod_pow(g, m, N*N)*mod_pow(r, N, N*N))%(N*N)

def D(c, pri_key, pub_key):
    lamb, mu = pri_key
    N, g = pub_key
    return ((mod_pow(c, lamb, N*N)-1)//N*mu)%N

def enc(img, pub_key):
    N, g = pub_key
    h, w = img.shape
    enc_img = [ [0 for j in range(w)] for i in range(h)]
    for i in range(h):
        for j in range(w//2):
            while True:
                r = random.randint(1, N*N)
                if math.gcd(r, N)==1:
                    break
            enc_img[i][2*j] = E(img[i][2*j], pub_key, r)
            enc_img[i][2*j+1] = E(img[i][2*j+1], pub_key, r)
            
    return enc_img

def emb(img, bitstr, pub_key):
    N, g = pub_key
    h = len(img)
    w = len(img[0])
    emb_img = [ [0 for j in range(w)] for i in range(h)]
    crypt_list = []
    for i in range(256):
        crypt_list.append(mod_pow(g, i, N*N))
    diff = np.zeros((h, w//2))
    pos_diff = {}
    neg_diff = {}
    max_embed = 0
    for i in range(h):
        for j in range(w//2):
            inv_1 = Crypto.Util.number.inverse(img[i][2*j], N*N)
            inv_2 = Crypto.Util.number.inverse(img[i][2*j+1], N*N)
            diff_1 = (img[i][2*j]*inv_2)%(N*N)
            diff_2 = (img[i][2*j+1]*inv_1)%(N*N)
            if diff_1 == 1:
                continue
            for n in range(1, 256):
                if diff_1 == crypt_list[n]:
                    diff[i][j] = n
                    if n in pos_diff.keys():
                        pos_diff[n] += 1
                    else:
                        pos_diff[n] = 1
                    break
                if diff_2 == crypt_list[n]:
                    diff[i][j] = -n
                    if -n in neg_diff.keys():
                        neg_diff[-n] += 1
                    else:
                        neg_diff[-n] = 1
                    break
            
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
            if diff[i][j] >= 0:
                if diff[i][j] > ec_pos:
                    emb_img[i][2*j] = (img[i][2*j]*E(1, pub_key))%(N*N)
                if diff[i][j] < ec_pos:
                    emb_img[i][2*j] = (img[i][2*j]*E(0, pub_key))%(N*N)
                if diff[i][j] == ec_pos:
                    if cnt < embed_len:
                        emb_img[i][2*j] = (img[i][2*j]*E(int(bitstr[cnt]), pub_key))%(N*N)
                        cnt += 1
                    else:
                        emb_img[i][2*j] = (img[i][2*j]*E(0, pub_key))%(N*N)
                emb_img[i][2*j+1] = (img[i][2*j+1]*E(0, pub_key))%(N*N)
            else:
                if diff[i][j] < ec_neg:
                    emb_img[i][2*j+1] = (img[i][2*j+1]*E(1, pub_key))%(N*N)
                if diff[i][j] > ec_neg:
                    emb_img[i][2*j+1] = (img[i][2*j+1]*E(0, pub_key))%(N*N)
                if diff[i][j] == ec_neg:
                    if cnt < embed_len:
                        emb_img[i][2*j+1] = (img[i][2*j+1]*E(int(bitstr[cnt]), pub_key))%(N*N)
                        cnt += 1
                    else:
                        emb_img[i][2*j+1] = (img[i][2*j+1]*E(0, pub_key))%(N*N)
                emb_img[i][2*j] = (img[i][2*j]*E(0, pub_key))%(N*N)

    return emb_img, ec_pos, ec_neg, embed_len

def dec(img, pri_key, pub_key):
    h = len(img)
    w = len(img[0])
    dec_img = [ [0 for j in range(w)] for i in range(h)]
    for i in range(h):
        for j in range(w//2):
            dec_img[i][2*j] = D(img[i][2*j], pri_key, pub_key)
            dec_img[i][2*j+1] = D(img[i][2*j+1], pri_key, pub_key)
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

    security_len = 24
    p = Crypto.Util.number.getPrime(security_len)
    q = Crypto.Util.number.getPrime(security_len)
    pri_key, pub_key = key_gen(p, q)

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

    image = dec(image, pri_key, pub_key)
    img = np.array(image, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('lena_decrypt.png', img)
    
    image, bit_str = ext(image, pos, neg, length)
    img = np.array(image, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('lena_restore.png', img)
    print(bits2str(bit_str))