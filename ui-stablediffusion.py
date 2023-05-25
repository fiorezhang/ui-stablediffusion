# -*- coding: UTF-8 -*-
 
from tkinter import * 
from tkinter.ttk import Progressbar
import ttkbootstrap
from ttkbootstrap.constants import *
import time
import random
from PIL import ImageTk,Image
import queue
from translate import translateYouDao
import numpy as np
from io import BytesIO
import win32clipboard
import os
import csv

# version info
VERSION = 'v2.2'

# You known you're on which platform, may change to auto-adapt in the future
PLATFORM = "I"
if PLATFORM == "I":
    from arc_ov import downloadModel, compileModel, generateImage
elif PLATFORM == "N":
    from cuda_py import downloadModel, compileModel, generateImage

# whether use youdao transfer
TRANSFER = True

# max images in preview
MAX_LIST_GENERATED_IMAGES = 9

# resolutions
RES_ORIGINAL = 512
RES_PREVIEW  = 64
RES_LOCK     = 192
RES_MASK     = 64

# define log file
FILE_LOG = "./perflog.csv"

# special settings for chilloutmix model(human portrait)
CHILL_MODEL = True
# some prompts for random generation
if CHILL_MODEL:
    PROMPT_LIST = ['a little girl with a red hat watching stars', 
    'a beautiful girl with silver earrings in a forest', 
    'a handsome man with dark t-shirt', 
    'a lovely boy with glasses']
    NEGATIVE_PROMPT_LIST = ['EasyNegative,paintings,sketches,low quality,grayscale,urgly face,extra fingers,fewer fingers,watermark']
else:
    PROMPT_LIST = ["The number 2 in leaves on a teal background",
    "colorful geometric shapes on white, graphic design",
    "A bridal bouquet made of succulents",
    "mountains, fantasy, digital art",
    "Bright eye makeup looks",
    "a cute shiba inu astronaut in a red and white space suit, yellow background, digital art",
    "robot made of analog stereo equipment, light purple background, digital art",
    "Futuristic scene with skyscrapers and hovercrafts, digital art",
    "lemon surrealist painting",
    "minimalist quartz and gold ring concept",
    "A renaissance  painting of a giraffe wearing a puff jacket",
    "Woman at misty galactic portal, 4k digital art",
    "Boho interior design with red accents",
    "cute polar bear, soft lighting and background, 100mm, blender render",
    "Thinly sliced cucumbers with salmon, piped cream cheese, and beluga caviar",
    "An abandoned medieval castle, impressionist painting",
    "Indigo dahlia macro",
    "Fairytale village inside a coconut",
    "A tall stack of pancakes, cropped food photography",
    "Dream alpine treehouse with sweeping mountain views",
    "a guitar made of flowers, light green background, digital art",
    "Astronaut skater in space nebula",
    "Strawberry, pop art",
    "Volcano exploding at dawn"]

    # negative prompts for reference
    NEGATIVE_PROMPT_LIST = ["malformed limbs",
    "out of frame",
    "signature",
    "blurry",
    "deformed",
    "fused fingers",
    "mutation",
    "duplicate",
    "extra fingers",
    "gross proportions",
    "mutilated",
    "watermark",
    "dehydrated",
    "jpeg artifacts",
    "username",
    "missing arms",
    "low quality",
    "bad proportions",
    "extra legs",
    "poorly drawn hands",
    "cropped",
    "lowres",
    "ugly",
    "too many fingers",
    "disfigured",
    "error",
    "extra limbs",
    "missing legs",
    "bad anatomy",
    "morbid",
    "long neck",
    "text",
    "extra arms",
    "mutated hands",
    "cloned face",
    "poorly drawn face",
    "out of frame",
    "worst quality"]
    
def appendCsv(file, row):
    try:
        with open(file, mode='a', newline='', encoding='utf-8-sig') as f:
            write = csv.writer(f)
            write.writerow(row)
            f.close
    except:
        pass

class UiStableDiffusion():
    def __init__(self):
        self.style = ttkbootstrap.Style(theme='superhero') # DARK- solar, superhero, darkly, cyborg, vapor; LIGHT- cosmo, flatly, journal, litera, lumen, minty, pulse, sandstone, united, yeti, morph, simplex, cerculean
        self.root = self.style.master
        self.root.geometry('1080x675+100+100')
        self.root.resizable(False, False)
        self.root.title('Stable Diffusion Demo ' + VERSION)
        self.root.overrideredirect(False)

        # create multiple Frames
        self.leftFrame = ttkbootstrap.Frame(self.root, width=250, height=540)
        self.middleFrame = ttkbootstrap.Frame(self.root, width=540, height=540)
        self.bottomFrame = ttkbootstrap.Frame(self.root, width=800, height=95)
        self.rightFrame = ttkbootstrap.Frame(self.root, width=260, height=645)

        self.leftFrame.grid(row=0, column=0, padx=5, pady=5)
        self.leftFrame.grid_propagate(False)
        self.middleFrame.grid(row=0, column=1, padx=5, pady=5)
        self.middleFrame.grid_propagate(False)
        self.bottomFrame.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.bottomFrame.grid_propagate(False)
        self.rightFrame.grid(row=0, column=2, rowspan=2, padx=5, pady=5)
        self.rightFrame.grid_propagate(False)

        # ====    locate user input in left Frame
        self.promptLabel = ttkbootstrap.Label(self.leftFrame, text='Prompt')
        self.promptButton = ttkbootstrap.Button(self.leftFrame, text='🎲', command=self.promptCallback, bootstyle=(LIGHT, OUTLINE))
        self.promptText = ttkbootstrap.Text(self.leftFrame, width=32, height=5)    
        self.negativeLabel = ttkbootstrap.Label(self.leftFrame, text='Negative Prompt')
        self.negativeButton = ttkbootstrap.Button(self.leftFrame, text='🎲', command=self.negativeCallback, bootstyle=(LIGHT, OUTLINE))
        self.negativeText = ttkbootstrap.Text(self.leftFrame, width=32, height=3)
        self.seedLabel = ttkbootstrap.Label(self.leftFrame, text='Seed')
        self.seedButton = ttkbootstrap.Button(self.leftFrame, text='🎲', command=self.seedCallback, bootstyle=(LIGHT, OUTLINE))
        self.seedEntry = ttkbootstrap.Entry(self.leftFrame, width=32)
        self.stepsLabel = ttkbootstrap.Label(self.leftFrame, text='Steps')
        self.stepsButton = ttkbootstrap.Button(self.leftFrame, text='🎲', command=self.stepsCallback, bootstyle=(LIGHT, OUTLINE))
        self.stepsEntry = ttkbootstrap.Entry(self.leftFrame, width=32)
        self.initializeButton = ttkbootstrap.Button(self.leftFrame, text='Initialize', command=self.initializeCallback, bootstyle=(PRIMARY, OUTLINE))
        self.generateButton = ttkbootstrap.Button(self.leftFrame, text='Generate', command=self.generateCallback, bootstyle=(PRIMARY, OUTLINE))
        
        self.promptLabel.grid(row=0, column=0, padx=5, pady=5)
        self.promptButton.grid(row=0, column=1, padx=5, pady=5)
        self.promptText.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.negativeLabel.grid(row=2, column=0, padx=5, pady=5)
        self.negativeButton.grid(row=2, column=1, padx=5, pady=5)
        self.negativeText.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        self.seedLabel.grid(row=4, column=0, padx=5, pady=5)
        self.seedButton.grid(row=4, column=1, padx=5, pady=5)
        self.seedEntry.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        self.stepsLabel.grid(row=6, column=0, padx=5, pady=5)
        self.stepsButton.grid(row=6, column=1, padx=5, pady=5)
        self.stepsEntry.grid(row=7, column=0, columnspan=2, padx=5, pady=5)
        self.initializeButton.grid(row=8, column=0, padx=5, pady=20)
        self.generateButton.grid(row=8, column=1, padx=5, pady=20)
                        
        # ====    locate main canvas in middle Frame
        self.canvasImage = ImageTk.PhotoImage(Image.open('ui/ui-welcome.png'))
        self.canvasLabel = ttkbootstrap.Label(self.middleFrame, image=self.canvasImage)
        self.canvasLabel.grid(row=0, column=0, padx=14, pady=14)
        
        self.canvasLabel.bind("<Double-Button-1>", self.copyToClipboard)

        # ====    locate config, time, progress bar in bottom Frame
        self.listPlatform = [('INTEL ', 0), ('NVIDIA', 1)]
        self.vPlatform = IntVar()
        if PLATFORM == "I":
            self.vPlatform.set(0)
        elif PLATFORM == "N":
            self.vPlatform.set(1)
        
        '''
        for platform, num in listPlatform:
            platformRadiobutton = Radiobutton(bottomFrame, text=platform, variable=vPlatform, value=num, width=10, indicatoron=False)
            platformRadiobutton.grid(row=0, column=num, padx=5, pady=0)    
        '''

        self.listXpu = [('CPU ', 0), ('GPU0', 1), ('GPU1', 2), ('AUTO', 3)]
        self.vXpu = IntVar()
        self.vXpu.set(3)
        
        for xpu, num in self.listXpu:
            self.xpuRadiobutton = Radiobutton(self.bottomFrame, text=xpu, variable=self.vXpu, value=num, width=6, indicatoron=False)
            self.xpuRadiobutton.grid(row=1, column=num, padx=5, pady=10)

        self.notifyLabel = ttkbootstrap.Label(self.bottomFrame, text='Re-Initialize after CPU/GPU Switch.', bootstyle=WARNING)    
        self.timeLabel = ttkbootstrap.Label(self.bottomFrame, text='Time:                     ', width=30)
        self.generateProgressbar = ttkbootstrap.Progressbar(self.bottomFrame, length=280, style='secondary.Striped.Horizontal.TProgressbar')
        self.generateAllProgressbar = ttkbootstrap.Progressbar(self.bottomFrame, length=280, style='secondary.Striped.Horizontal.TProgressbar')

        self.notifyLabel.grid(row=2, column=0, columnspan=3, padx=5, pady=10)    
        self.timeLabel.grid(row=1, column=4, rowspan=2, padx=20, pady=10)
        self.generateProgressbar.grid(row=1, column=5, padx=5, pady=10)
        self.generateAllProgressbar.grid(row=2, column=5, padx=5, pady=10)

        # ====    locate preview and input image in right Frame
        self.listGeneratedImages = []   #generated images, can be many
        self.listGeneratedCanvas = []   #generated canvas for preview, fixed
        self.vGeneratedImage = IntVar()
        self.vGeneratedImage.set(0)
        
        self.previewLabel = ttkbootstrap.Label(self.rightFrame, text='Preview')
        for indexImage in range(MAX_LIST_GENERATED_IMAGES):
            self.canvasGeneratedImage = ImageTk.PhotoImage(Image.open('ui/ui-blank.png').resize((RES_PREVIEW, RES_PREVIEW)))
            self.imageRadiobutton = Radiobutton(self.rightFrame, image=self.canvasGeneratedImage, variable=self.vGeneratedImage, value=indexImage, width=RES_PREVIEW, height=RES_PREVIEW, indicatoron=False)
            self.imageRadiobutton.grid(row=1+int(indexImage/3), column=int(indexImage%3), padx=5, pady=5)
            self.listGeneratedCanvas.append({'button':self.imageRadiobutton, 'image':self.canvasGeneratedImage})
            
        self.lockButton = ttkbootstrap.Button(self.rightFrame, text='Lock ↓↓', command=self.lockCallback, bootstyle=(LIGHT, OUTLINE))
        self.unlockButton = ttkbootstrap.Button(self.rightFrame, text='Unlock', command=self.unlockCallback, bootstyle=(LIGHT, OUTLINE))
        self.inputLabel = ttkbootstrap.Label(self.rightFrame, text='Input image')       
        self.canvasLockedCanvas = ttkbootstrap.Canvas(self.rightFrame, width=RES_LOCK, height=RES_LOCK)
        self.canvasLockedImage = ImageTk.PhotoImage(Image.open('ui/ui-blank.png').resize((RES_LOCK, RES_LOCK)))
        self.canvasLockedCanvas.create_image(0, 0, anchor=NW, image=self.canvasLockedImage)
        self.noiseLabel = ttkbootstrap.Label(self.rightFrame, text='Variation Ratio: 0.1-1.0') 
        self.noiseScale = ttkbootstrap.Scale(self.rightFrame, from_=1, to=10, orient=HORIZONTAL)
        self.noiseScale.set(5)
        self.maskLabel = ttkbootstrap.Label(self.rightFrame, text='Inpaint masked')
        self.clearMaskButton = ttkbootstrap.Button(self.rightFrame, text='Clear <<', command=self.clearMaskCallback, bootstyle=(LIGHT, OUTLINE))
        self.backMaskButton = ttkbootstrap.Button(self.rightFrame, text='  Back <', command=self.backMaskCallback, bootstyle=(LIGHT, OUTLINE))
        
        self.previewLabel.grid(row=0, column=0, columnspan=3, padx=5, pady=5)
        self.lockButton.grid(row=1+round(MAX_LIST_GENERATED_IMAGES/3), column=0, padx=5, pady=5)
        self.unlockButton.grid(row=1+round(MAX_LIST_GENERATED_IMAGES/3), column=2, padx=5, pady=5)
        self.inputLabel.grid(row=1+round(MAX_LIST_GENERATED_IMAGES/3), column=1, padx=5, pady=5)
        self.canvasLockedCanvas.grid(row=2+round(MAX_LIST_GENERATED_IMAGES/3), column=0, columnspan=3, padx=5, pady=5)
        self.noiseLabel.grid(row=3+round(MAX_LIST_GENERATED_IMAGES/3), column=0, columnspan=3, padx=5, pady=5)
        self.noiseScale.grid(row=4+round(MAX_LIST_GENERATED_IMAGES/3), column=0, columnspan=3, padx=5, pady=5)
        self.maskLabel.grid(row=5+round(MAX_LIST_GENERATED_IMAGES/3), column=1, padx=5, pady=5)
        self.clearMaskButton.grid(row=5+round(MAX_LIST_GENERATED_IMAGES/3), column=0, padx=5, pady=5)
        self.backMaskButton.grid(row=5+round(MAX_LIST_GENERATED_IMAGES/3), column=2, padx=5, pady=5)
        
        self.canvasLockedCanvas.bind('<Button-1>', self.getMaskStartInfo)
        self.canvasLockedCanvas.bind('<B1-Motion>', self.getMaskMidInfo)
        self.canvasLockedCanvas.bind('<ButtonRelease-1>', self.getMaskEndInfo)
        
        #==========================
        # queue for generation tasks    
        self.queueTaskGenerate = queue.Queue()
        # record last selected image index in preview
        self.lastGeneratedImageIndex = 0   
        # local ov pipe instance
        self.localPipe = None
        # main image file path
        self.mainImageFile = ""
        # locked image file path, "" means unlocked
        self.lockedImageFile = ""        
        # mask coordination
        self.startX = 0
        self.startY = 0
        self.midX   = 0
        self.midY   = 0
        self.endX   = 0
        self.endY   = 0
        self.listMaskRect = []
                
        if not os.path.exists(FILE_LOG): 
            rowFirstLine = ["Timestamp", "TimeCost", "Steps", "Seed", "Prompt", "Negative"]
            appendCsv(FILE_LOG, rowFirstLine)
                
        #==========================
        # kick off async thread
        self.root.after(100, self.asyncLoopGenerate)
        
        # kick off main loop
        self.root.mainloop()              

    # actions to buttons
    def initializeCallback(self):        
        platformIndex = self.vPlatform.get()
        if PLATFORM == "I" and platformIndex == 0:  #Intel
            pass
        elif PLATFORM == "N" and platformIndex == 1: #Nvidia
            pass
        else:
            self.canvasImage = ImageTk.PhotoImage(Image.open('ui/ui-compatibility.png'))
            self.canvasLabel.configure(image=self.canvasImage)    
            return
        
        downloadModel()

        xpuIndex = self.vXpu.get()
        if xpuIndex == 0: #CPU
            self.localPipe = compileModel('CPU')
            self.canvasImage = ImageTk.PhotoImage(Image.open('ui/ui-ready.png'))
            self.canvasLabel.configure(image=self.canvasImage)  
        elif xpuIndex == 1: #GPU 0
            self.localPipe = compileModel('GPU.0')
            self.canvasImage = ImageTk.PhotoImage(Image.open('ui/ui-ready.png'))
            self.canvasLabel.configure(image=self.canvasImage)  
        elif xpuIndex == 2: #GPU 1
            self.localPipe = compileModel('GPU.1')
            self.canvasImage = ImageTk.PhotoImage(Image.open('ui/ui-ready.png'))
            self.canvasLabel.configure(image=self.canvasImage)     
        elif xpuIndex == 3: #AUTO
            self.localPipe = compileModel('AUTO')
            self.canvasImage = ImageTk.PhotoImage(Image.open('ui/ui-ready.png'))
            self.canvasLabel.configure(image=self.canvasImage) 

        # clear progressbar
        self.generateProgressbar['value'] = 0
        self.generateAllProgressbar['value'] = 0
        # clear generated images
        self.listGeneratedImages.clear()
        self.clearGeneratedImages()
        self.mainImageFile = ""
        self.lockedImageFile = ""
        self.unlockCallback()

    def promptCallback(self):
        self.promptText.delete('1.0', END)
        self.promptText.insert('1.0', random.choices(PROMPT_LIST)[0])
        
    def negativeCallback(self):
        #negativeText.delete('1.0', END)
        self.negativeText.insert('1.0', random.choices(NEGATIVE_PROMPT_LIST)[0]+',')
      
    def seedCallback(self):
        self.seedEntry.delete(0, END)
        self.seedEntry.insert(0, random.randint(0, 9999))
        
    def stepsCallback(self):
        self.stepsEntry.delete(0, END)
        self.stepsEntry.insert(0, random.randint(20, 50))

    def sendMsgToClipboard(self, typeData, msg):
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(typeData, msg)
        win32clipboard.CloseClipboard()
  
    def pasteToClipboard(self, file):
        image = Image.open(file)
        output = BytesIO()
        image.convert('RGB').save(output, 'BMP')
        data = output.getvalue()[14:]
        output.close()
        self.sendMsgToClipboard(win32clipboard.CF_DIB, data)
        
    def copyToClipboard(self, event):
        self.pasteToClipboard(self.mainImageFile)
        print("Copy to clipboard")
        
    def lockCallback(self):
        if self.generateButton['text'] == 'Generate':    
            currentGeneratedImageIndex = self.vGeneratedImage.get()
            try:
                self.lockedImageFile = self.listGeneratedImages[currentGeneratedImageIndex]
                self.canvasLockedImage = ImageTk.PhotoImage(Image.open(self.lockedImageFile).resize((RES_LOCK, RES_LOCK)))
                self.canvasLockedCanvas.delete("all")
                self.clearMaskCallback()
                self.canvasLockedCanvas.create_image(0, 0, anchor=NW, image=self.canvasLockedImage)
            except:
                pass

    def unlockCallback(self):
        if self.generateButton['text'] == 'Generate':    
            self.lockedImageFile = ""
            self.canvasLockedImage = ImageTk.PhotoImage(Image.open('ui/ui-blank.png').resize((RES_LOCK, RES_LOCK)))
            self.canvasLockedCanvas.delete("all")
            self.clearMaskCallback()
            self.canvasLockedCanvas.create_image(0, 0, anchor=NW, image=self.canvasLockedImage)

    def clip(self, v, vmin, vmax):
        if v < vmin:
            v = vmin
        elif v > vmax:
            v = vmax
        return v

    def getMaskStartInfo(self, event):
        self.startX = self.clip(event.x, 0, RES_LOCK-1)
        self.startY = self.clip(event.y, 0, RES_LOCK-1)
        
    def getMaskMidInfo(self, event):
        self.midX = self.clip(event.x, 0, RES_LOCK-1)
        self.midY = self.clip(event.y, 0, RES_LOCK-1)
        self.canvasLockedCanvas.delete("tempRect")
        idMaskTempRect = self.canvasLockedCanvas.create_rectangle(self.startX, self.startY, self.midX, self.midY, fill='', outline='black', tags='tempRect')
        
    def getMaskEndInfo(self, event):
        self.endX = self.clip(event.x, 0, RES_LOCK-1)
        self.endY = self.clip(event.y, 0, RES_LOCK-1)
        self.canvasLockedCanvas.delete("tempRect")
        idMaskRect = self.canvasLockedCanvas.create_rectangle(self.startX, self.startY, self.endX, self.endY, fill='black', outline='black', tags='maskRect')
        self.listMaskRect.append({"startX": self.startX, "startY": self.startY, "endX": self.endX, "endY": self.endY, "id": idMaskRect})

    def clearMaskCallback(self):
        self.listMaskRect.clear()
        self.canvasLockedCanvas.delete("maskRect")
    
    def backMaskCallback(self):
        if len(self.listMaskRect) > 0:
            lastMaskRect = self.listMaskRect.pop()
            if lastMaskRect is not None:
                self.canvasLockedCanvas.delete(lastMaskRect["id"])

    def handleMetaString(self, userinput, vmin=1, vmax=100):
        output = []
        pieceList = userinput.split(',')
        for piece in pieceList:
            if '_' in piece:
                numList = piece.split('_')
                if len(numList) == 2 and int(numList[0]) <= int(numList[1]):
                    for i in range(int(numList[0]), int(numList[1])+1):
                        output.append(i)
            elif '+' in piece:
                numList = piece.split('+')
                if len(numList) == 2:
                    vnum = int(numList[0]) if int(numList[0]) != -1 else random.randint(vmin, vmax)
                    for i in range(vnum, vnum+int(numList[1])):
                        output.append(i)
            else:
                vnum = int(piece) if int(piece) != -1 else random.randint(vmin, vmax)
                output.append(vnum)
        return output

    def generateCallback(self):
        # async routine, when generate clicked, change the 'text' to 'interrupt', when next click, interrupt current unfinished task queue
        if self.generateButton['text'] == 'Generate':   
            # read parameters for text -> image
            prompt = self.promptText.get('1.0', END).replace('\n', '').replace('\t', '')
            if TRANSFER:
                prompt = translateYouDao(prompt)
            negative = self.negativeText.get('1.0', END).replace('\n', '').replace('\t', '')
            seed = self.seedEntry.get()
            steps = self.stepsEntry.get()
            # get input image and strenth for image -> image
            image = self.lockedImageFile
            strength = self.noiseScale.get() / 10
            # get mask for inpaint(enhanced image -> image)
            if len(self.listMaskRect) > 0:
                mask = np.ones((1, 4, RES_MASK, RES_MASK))
                for maskRect in self.listMaskRect:
                    startX, startY, endX, endY = maskRect["startX"], maskRect["startY"], maskRect["endX"], maskRect["endY"]
                    startX, startY, endX, endY = round(startX/RES_LOCK*RES_MASK), round(startY/RES_LOCK*RES_MASK), round(endX/RES_LOCK*RES_MASK), round(endY/RES_LOCK*RES_MASK)
                    if startX > endX:
                        startX, endX = endX, startX
                    if startY > endY:
                        startY, endY = endY, startY
                    mask[:,:,startY:endY,startX:endX]=0
            else:
                mask = None
        
            try:
                seedList = self.handleMetaString(seed, 0, 9999)
                stepsList = self.handleMetaString(steps, 20, 50)
                
                self.generateAllProgressbar['maximum'] = sum(stepsList) * len(seedList)
                self.generateAllProgressbar['value'] = 0
                
                for steps in stepsList:
                    for seed in seedList:
                        xpuIndex = self.vXpu.get()
                        if xpuIndex == 0: #CPU
                            taskGenerate = ['CPU', prompt, negative, seed, steps, image, strength, mask]
                        elif xpuIndex == 1: #GPU 0
                            taskGenerate = ['GPU.0', prompt, negative, seed, steps, image, strength, mask]
                        elif xpuIndex == 2: #GPU 1
                            taskGenerate = ['GPU.1', prompt, negative, seed, steps, image, strength, mask]
                        elif xpuIndex == 3: #AUTO
                            taskGenerate = ['AUTO', prompt, negative, seed, steps, image, strength, mask]
                        self.queueTaskGenerate.put(taskGenerate)
                        
                self.generateButton['text'] = 'Interrupt'
            except NameError:
                self.canvasImage = ImageTk.PhotoImage(Image.open('ui/ui-initialize.png'))
                self.canvasLabel.configure(image=self.canvasImage)
            except ValueError:
                self.canvasImage = ImageTk.PhotoImage(Image.open('ui/ui-input.png'))
                self.canvasLabel.configure(image=self.canvasImage) 
        # async routine, when in last generation batch, the button shows 'interrupt', can't start next batch. click the button to clear the queue
        elif self.generateButton['text'] == 'Interrupt':
            self.queueTaskGenerate.queue.clear()
            #generateProgressbar['value'] = 0
            #generateAllProgressbar['value'] = 0

    def insertGeneratedImage(self, file):
        if len(self.listGeneratedImages) == MAX_LIST_GENERATED_IMAGES:
            self.listGeneratedImages.pop(-1)
        self.listGeneratedImages.insert(0, file)
        
    def previewGeneratedImages(self):
        for indexImage, generatedImage in enumerate(self.listGeneratedImages):
            self.listGeneratedCanvas[indexImage]["image"] = ImageTk.PhotoImage(Image.open(generatedImage).resize((RES_PREVIEW, RES_PREVIEW)))
            self.listGeneratedCanvas[indexImage]["button"].configure(image=self.listGeneratedCanvas[indexImage]["image"])    

    def clearGeneratedImages(self):
        for indexImage in range(MAX_LIST_GENERATED_IMAGES):
            self.listGeneratedCanvas[indexImage]["image"] = ImageTk.PhotoImage(Image.open('ui/ui-blank.png').resize((RES_PREVIEW, RES_PREVIEW)))
            self.listGeneratedCanvas[indexImage]["button"].configure(image=self.listGeneratedCanvas[indexImage]["image"])    

    # Create a loop here to async generate images - unblock mainloop windows message management
    # when clicked button, quickly exit the response function there, then the loop routine get chance to get in
    # in this loop routine, if we do a while to draw images, the windows messsage will be blocked then we still can't see all
    # we have to draw one image then return current loop routine, draw next in next loop, then everything perfect    
    def progressbarCallback(self):
        self.generateProgressbar['value'] = self.generateProgressbar['value'] + 1
        self.generateAllProgressbar['value'] = self.generateAllProgressbar['value'] + 1
        self.root.update()
             
    def asyncLoopGenerate(self):
        intervalLoop = 100 #ms
        if not self.queueTaskGenerate.empty():   # in generation work
            taskGenerate = self.queueTaskGenerate.get()
            try:
                startTime = time.time()
                xpu, prompt, negative, seed, steps, image, strength, mask = taskGenerate
                self.generateProgressbar['maximum'] = steps
                self.generateProgressbar['value'] = 0
                
                if self.localPipe == None:
                    print("Error: null local pipe!")
                else:
                    result = generateImage(xpu, self.localPipe, prompt, negative, seed, steps, image, strength, mask, self.progressbarCallback)
                self.mainImageFile = result
                self.canvasImage = ImageTk.PhotoImage(Image.open(self.mainImageFile).resize((RES_ORIGINAL, RES_ORIGINAL)))
                self.canvasLabel.configure(image=self.canvasImage)
                endTime = time.time()
                useTime = endTime-startTime
                roughSteps = int(steps) if image == "" else int(steps)*float(strength)
                rowLog = [time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime()), str(useTime), str(roughSteps), str(seed), str(prompt), str(negative)]
                appendCsv(FILE_LOG, rowLog)
                self.timeLabel.configure(text='Time: ' + "%.2f"%useTime + 's (' + "%.2f"%(roughSteps/useTime) + 'it/s)')   
                self.insertGeneratedImage(self.mainImageFile)
                self.vGeneratedImage.set(0)
                self.previewGeneratedImages()
            except NameError:
                self.canvasImage = ImageTk.PhotoImage(Image.open('ui/ui-initialize.png'))
                self.canvasLabel.configure(image=self.canvasImage)
                self.generateButton['text'] = 'Generate'
            except ValueError:
                self.canvasImage = ImageTk.PhotoImage(Image.open('ui/ui-input.png'))
                self.canvasLabel.configure(image=self.canvasImage)   
                self.generateButton['text'] = 'Generate'
        else:   # not in generation work, free for next work
            self.generateButton['text'] = 'Generate'
            # show selected image to main canvas
            currentGeneratedImageIndex = self.vGeneratedImage.get()
            if  currentGeneratedImageIndex!= self.lastGeneratedImageIndex:
                try:
                    self.mainImageFile = self.listGeneratedImages[currentGeneratedImageIndex]
                    self.canvasImage = ImageTk.PhotoImage(Image.open(self.mainImageFile).resize((RES_ORIGINAL, RES_ORIGINAL)))
                    self.canvasLabel.configure(image=self.canvasImage)
                    self.lastGeneratedImageIndex = currentGeneratedImageIndex
                except:
                    pass
        # iterately call next routine
        self.root.after(intervalLoop, self.asyncLoopGenerate)

# ### MAIN
#   
if __name__ == "__main__":       
    UiStableDiffusion()