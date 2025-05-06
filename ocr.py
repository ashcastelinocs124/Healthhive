
def opencv_format(self):
    pil_image = Image.open(imagepath).convert("RGB")
    numpy_image = np.array(pil_image)
    opecv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opecv_image
  
def grayscale(self, image = None):#Binarization
    if image is None:
      image = self.opencv_format()
    img = Image.open(imagepath).convert('L')
    return img
  
def threshold(self):
    image = self.grayscale()
    img = cv2.imread(imagepath, 0)
    text = pytesseract.image_to_string(image)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    print(text)
    #Value of a pixel is less then zero, we make it black if it is more then zero we will convert it to 1
    return 0
