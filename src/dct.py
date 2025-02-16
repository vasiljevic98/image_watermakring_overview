from scipy.fftpack import dct, idct

def dct2(a):
    return dct( dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return idct( idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')


#C(4,1) >= C(2,3) -> read as bit 0
#C(4,1) < C(2,3) -> read as bit 1
def dct_write8x8(img,  bit, c1 = [3,0], c2 = [1,2]):
    dct_img = dct2(img)
    if((dct_img[c1] >= dct_img[c2] and bit == 1) or (dct_img[c1] < dct_img[c2] and bit == 0)):
        dct_img[c1], dct_img[c2] = dct_img[c2], dct_img[c1]
    return idct2(dct_img)

       
      

 