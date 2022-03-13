import threading
import time
import params as p
import Evolve as EV
from PIL import Image



image = Image.open(p.image)
pixels = list(image.getdata())
(sizeX, sizeY) = (image.size[0],image.size[1])

redP = []
blueP = []
greenP = []

for pixel in pixels:
   redP.append(pixel[0])
   greenP.append(pixel[1])
   blueP.append(pixel[2])


#EV.runs(in_,out_,"red")

exitFlag = 0

class myThread (threading.Thread):
   def __init__(self, in_, sizeX, sizeY, colour):
      threading.Thread.__init__(self)
      self.in_ = in_
      self.sizeX = sizeX
      self.sizeY = sizeY
      self.colour = colour
   def run(self):
      print ("Starting " + self.name)
      EV.runs(self.in_, self.sizeX, self.sizeY, self.colour)
      print ("Exiting " + self.name)

# Create new threads
R = myThread(redP, sizeX, sizeY, "red")
G = myThread(greenP, sizeX, sizeY, "green")
B = myThread(blueP, sizeX, sizeY, "blue")

# Start new Threads
R.start()
G.start()
B.start()
R.join()
G.join()
B.join()
print ("Exiting Main Thread")


rTree = open("red.txt",'r')
gTree = open("green.txt",'r')
bTree = open("blue.txt",'r')

print(rTree.readline())
print(gTree.readline())
print(bTree.readline())

