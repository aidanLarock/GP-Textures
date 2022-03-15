import threading
import params as p
import Evolve as EV
from PIL import Image
import operator

from deap import gp


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


rTree = open("./TreeGenerations/red.txt",'r')
gTree = open("./TreeGenerations/green.txt",'r')
bTree = open("./TreeGenerations/blue.txt",'r')

rTree = rTree.readline()
gTree = gTree.readline()
bTree = bTree.readline()

def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

pset = gp.PrimitiveSet("MAIN", p.terminals)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)

#pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')


treeR = gp.compile(rTree,pset)
treeG = gp.compile(gTree,pset)
treeB = gp.compile(bTree,pset)

input_image = Image.new(mode="RGB", size=(sizeX, sizeY),
                     color="blue")

pixel_map = input_image.load()

for i in range(sizeX):
   for j in range(sizeY): 
      pixel_map[i,j] = (int(treeR(i,j)), int(treeG(i,j)), int(treeB(i,j)))

#input_image.show()
input_image.save("./NewImages/"+str(p.numGenerations)+
               "-"+str(p.popSize)+
               "-"+str(p.seed)+'.png')
