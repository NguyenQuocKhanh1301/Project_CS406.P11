import cv2

class convertImage:
  def __init__(self, image, coords):
    self.image  = image.copy()
    self.coords = coords

  

  def gaussianBlur(self, kernelSize):
    for x1, y1, x2, y2 in self.coords:
      sub = self.image[y1:y2, x1:x2]
      sub = cv2.GaussianBlur(sub, (kernelSize, kernelSize), 0)
      self.image[y1:y2, x1:x2] = sub
    return self.image

  def medianBlur(self, kernelSize):
    for x1, y1, x2, y2 in self.coords:
      sub = self.image[y1:y2, x1:x2]
      sub = cv2.medianBlur(sub, kernelSize)
      self.image[y1:y2, x1:x2] = sub
    return self.image


  def replaceImage(self, image):
    for x1, y1, x2, y2 in self.coords:
      sub = self.image[y1:y2, x1:x2]
      height, width = sub.shape[:2]
      self.image[y1:y2, x1:x2] = cv2.resize(image, (width, height))
    return self.image

  def getResult(self, typeBlur, kernelSize, kernelDepth, imageReplace):
    if typeBlur == 1:
        return self.gaussianBlur(kernelSize)
    elif typeBlur == 2:
        return self.medianBlur(kernelSize)
    return self.replaceImage(imageReplace)
