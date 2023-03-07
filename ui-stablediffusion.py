# -*- coding: UTF-8 -*-
 
from tkinter import * 
from tkinter.ttk import Progressbar
import ttkbootstrap
from ttkbootstrap.constants import *
import time
import random

VERSION = 'v1.3'

# You known you're on which platform, may change to auto-adapt in the future
PLATFORM = "I"

if PLATFORM == "I":
    from arc_ov import downloadModel, compileModel, generateImage, BETA_MODE
elif PLATFORM == "N":
    from cuda_py import downloadModel, compileModel, generateImage
    
if BETA_MODE:
    VERSION = VERSION + '-beta'
    import queue
    queueTaskGenerate = queue.Queue()

CHILL_MODEL = True
# some prompts for random generation
if CHILL_MODEL:
    promptList = ['a lovely girl with a red hat', 
    'a beautiful girl with silver earrings', 
    'a handsome man with dark t-shirt', 
    'a lovely boy with glasses']
    negativeList = ['EasyNegative,paintings,sketches,low quality,grayscale,urgly face,extra fingers,fewer fingers,watermark']
else:
    promptList = ["The number 2 in leaves on a teal background",
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
    negativeList = ["malformed limbs",
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


# actions to buttons
def initializeCallback():
    global localPipe, canvasImage, canvasLabel
    
    platformIndex = vPlatform.get()
    if PLATFORM == "I" and platformIndex == 0:  #Intel
        pass
    elif PLATFORM == "N" and platformIndex == 1: #Nvidia
        pass
    else:
        canvasImage = PhotoImage(file='ui/ui-compatibility.png')
        canvasLabel.configure(image=canvasImage)    
        return
    
    downloadModel()
        
    if BETA_MODE:
        xpuIndex = vXpu.get()
        if xpuIndex == 0: #CPU
            localPipe = compileModel('CPU')
            canvasImage = PhotoImage(file='ui/ui-ready.png')
            canvasLabel.configure(image=canvasImage)  
        elif xpuIndex == 1: #GPU 0
            localPipe = compileModel('GPU.0')
            canvasImage = PhotoImage(file='ui/ui-ready.png')
            canvasLabel.configure(image=canvasImage)  
        elif xpuIndex == 2: #GPU 1
            localPipe = compileModel('GPU.1')
            canvasImage = PhotoImage(file='ui/ui-ready.png')
            canvasLabel.configure(image=canvasImage)              
    else:
        xpuIndex = vXpu.get()
        if xpuIndex == 0: #CPU
            localPipe = compileModel('CPU')
            canvasImage = PhotoImage(file='ui/ui-ready.png')
            canvasLabel.configure(image=canvasImage)  
        elif xpuIndex == 1: #GPU
            localPipe = compileModel('GPU')
            canvasImage = PhotoImage(file='ui/ui-ready.png')
            canvasLabel.configure(image=canvasImage)  

def promptCallback():
    promptText.delete('1.0', END)
    promptText.insert('1.0', random.choices(promptList)[0])
    
def negativeCallback():
    if BETA_MODE: 
        negativeText.delete('1.0', END)
    negativeText.insert('1.0', random.choices(negativeList)[0]+',')
  
def seedCallback():
    seedEntry.delete(0, END)
    seedEntry.insert(0, random.randint(0, 999))
    
def stepsCallback():
    stepsEntry.delete(0, END)
    stepsEntry.insert(0, random.randint(20, 50))

def handleMetaString(userinput):
    output = []
    pieceList = userinput.split(',')
    for piece in pieceList:
        numList = piece.split('-')
        if len(numList) == 1:
            output.append(int(numList[0]))
        if len(numList) == 2:
            for i in range(int(numList[0]), int(numList[1])+1):
                output.append(i)
    return output

def generateCallback():
    prompt = promptText.get('1.0', END).replace('\n', '').replace('\t', '')
    negative = negativeText.get('1.0', END).replace('\n', '').replace('\t', '')
    seed = seedEntry.get()
    steps = stepsEntry.get()

    global localPipe, canvasImage, canvasLabel, generateAllProgressbar, generateButton
    if not BETA_MODE:        
        try:
            startTime = time.time()
            xpuIndex = vXpu.get()
            if xpuIndex == 0: #CPU
                imageCurrent = generateImage('CPU', localPipe, prompt, negative, seed, steps, callback=None)
            elif xpuIndex == 1: #GPU
                imageCurrent = generateImage('GPU', localPipe, prompt, negative, seed, steps, callback=None)
            canvasImage = PhotoImage(file=imageCurrent)
            canvasLabel.configure(image=canvasImage)           
            endTime = time.time()
            timeLabel.configure(text='Time: ' + "%.2f"%(endTime-startTime) + 's (' + "%.2f"%(int(steps)/(endTime-startTime)) + 'it/s)')        
        except NameError:
            canvasImage = PhotoImage(file='ui/ui-initialize.png')
            canvasLabel.configure(image=canvasImage)
        except ValueError:
            canvasImage = PhotoImage(file='ui/ui-input.png')
            canvasLabel.configure(image=canvasImage)
    else: #BETA_MODE
        # async routine, when generate clicked, change the 'text' to 'interrupt', when next click, interrupt current unfinished task queue
        if generateButton['text'] == 'Generate':    
            try:
                seedList = handleMetaString(seed)
                stepsList = handleMetaString(steps)
                
                generateAllProgressbar['maximum'] = sum(stepsList) * len(seedList)
                generateAllProgressbar['value'] = 0
                
                for steps in stepsList:
                    for seed in seedList:
                        xpuIndex = vXpu.get()
                        if xpuIndex == 0: #CPU
                            taskGenerate = ['CPU', localPipe, prompt, negative, seed, steps]
                        elif xpuIndex == 1: #GPU 0
                            taskGenerate = ['GPU.0', localPipe, prompt, negative, seed, steps]
                        elif xpuIndex == 2: #GPU 1
                            taskGenerate = ['GPU.1', localPipe, prompt, negative, seed, steps]
                        queueTaskGenerate.put(taskGenerate)
                        
                generateButton['text'] = 'Interrupt'
            except NameError:
                canvasImage = PhotoImage(file='ui/ui-initialize.png')
                canvasLabel.configure(image=canvasImage)
            except ValueError:
                canvasImage = PhotoImage(file='ui/ui-input.png')
                canvasLabel.configure(image=canvasImage) 
        # async routine, when in last generation batch, the button shows 'interrupt', can't start next batch. click the button to clear the queue
        elif generateButton['text'] == 'Interrupt':
            queueTaskGenerate.queue.clear()
            #generateProgressbar['value'] = 0
            #generateAllProgressbar['value'] = 0

# Create a loop here to async generate images - unblock mainloop windows message management
# when clicked button, quickly exit the response function there, then the loop routine get chance to get in
# in this loop routine, if we do a while to draw images, the windows messsage will be blocked then we still can't see all
# we have to draw one image then return current loop routine, draw next in next loop, then everything perfect    
def progressbarCallback():
    global root, generateProgressbar, generateAllProgressbar
    generateProgressbar['value'] = generateProgressbar['value'] + 1
    generateAllProgressbar['value'] = generateAllProgressbar['value'] + 1
    root.update()
                        
def asyncLoopGenerate():
    global root,canvasImage, canvasLabel, timeLabel, generateProgressbar, generateButton
    intervalLoop = 100 #ms
    if not queueTaskGenerate.empty():
        taskGenerate = queueTaskGenerate.get()
        try:
            startTime = time.time()
            xpu, localPipe, prompt, negative, seed, steps = taskGenerate
            generateProgressbar['maximum'] = steps
            generateProgressbar['value'] = 0
            
            imageCurrent = generateImage(xpu, localPipe, prompt, negative, seed, steps, progressbarCallback)
            canvasImage = PhotoImage(file=imageCurrent)
            canvasLabel.configure(image=canvasImage)
            endTime = time.time()
            timeLabel.configure(text='Time: ' + "%.2f"%(endTime-startTime) + 's (' + "%.2f"%(int(steps)/(endTime-startTime)) + 'it/s)')   
        except NameError:
            canvasImage = PhotoImage(file='ui/ui-initialize.png')
            canvasLabel.configure(image=canvasImage)
        except ValueError:
            canvasImage = PhotoImage(file='ui/ui-input.png')
            canvasLabel.configure(image=canvasImage)   
    else:
        generateButton['text'] = 'Generate'
    root.after(intervalLoop, asyncLoopGenerate)

# ### MAIN
#   
if __name__ == "__main__":    
    if BETA_MODE:
        style = ttkbootstrap.Style(theme='superhero') # DARK- solar, superhero, darkly, cyborg, vapor; LIGHT- cosmo, flatly, journal, litera, lumen, minty, pulse, sandstone, united, yeti, morph, simplex, cerculean
    else:
        style = ttkbootstrap.Style(theme='superhero') # DARK- solar, superhero, darkly, cyborg, vapor; LIGHT- cosmo, flatly, journal, litera, lumen, minty, pulse, sandstone, united, yeti, morph, simplex, cerculean
    root = style.master
    root.geometry('800x600+100+100')
    root.resizable(False, False)
    root.title('Stable Diffusion Demo ' + VERSION)
    root.overrideredirect(False)

    # 创建多个Frame
    leftFrame = ttkbootstrap.Frame(root, width=250, height=520)
    rightFrame = ttkbootstrap.Frame(root, width=530, height=520)
    bottomFrame = ttkbootstrap.Frame(root, width=790, height=50)

    leftFrame.grid(row=0, column=0, padx=5, pady=5)
    leftFrame.grid_propagate(False)
    rightFrame.grid(row=0, column=1, padx=5, pady=5)
    rightFrame.grid_propagate(False)
    bottomFrame.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
    bottomFrame.grid_propagate(False)

    # 在左侧Frame放置用户输入
    promptLabel = ttkbootstrap.Label(leftFrame, text='Prompt')
    promptButton = ttkbootstrap.Button(leftFrame, text='🎲', command=promptCallback, bootstyle=(LIGHT, OUTLINE))
    promptText = ttkbootstrap.Text(leftFrame, width=32, height=5)    
    negativeLabel = ttkbootstrap.Label(leftFrame, text='Negative Prompt')
    negativeButton = ttkbootstrap.Button(leftFrame, text='🎲', command=negativeCallback, bootstyle=(LIGHT, OUTLINE))
    negativeText = ttkbootstrap.Text(leftFrame, width=32, height=3)
    seedLabel = ttkbootstrap.Label(leftFrame, text='Seed')
    seedButton = ttkbootstrap.Button(leftFrame, text='🎲', command=seedCallback, bootstyle=(LIGHT, OUTLINE))
    seedEntry = ttkbootstrap.Entry(leftFrame, width=32)
    stepsLabel = ttkbootstrap.Label(leftFrame, text='Steps')
    stepsButton = ttkbootstrap.Button(leftFrame, text='🎲', command=stepsCallback, bootstyle=(LIGHT, OUTLINE))
    stepsEntry = ttkbootstrap.Entry(leftFrame, width=32)
    initializeButton = ttkbootstrap.Button(leftFrame, text='Initialize', command=initializeCallback, bootstyle=(PRIMARY, OUTLINE))
    generateButton = ttkbootstrap.Button(leftFrame, text='Generate', command=generateCallback, bootstyle=(PRIMARY, OUTLINE))
    notifyLabel = ttkbootstrap.Label(leftFrame, text='Re-Initialize after CPU/GPU Switching.', font=('helvetica', 8), bootstyle=WARNING)

    promptLabel.grid(row=0, column=0, padx=5, pady=5)
    promptButton.grid(row=0, column=1, padx=5, pady=5)
    promptText.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
    negativeLabel.grid(row=2, column=0, padx=5, pady=5)
    negativeButton.grid(row=2, column=1, padx=5, pady=5)
    negativeText.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
    seedLabel.grid(row=4, column=0, padx=5, pady=5)
    seedButton.grid(row=4, column=1, padx=5, pady=5)
    seedEntry.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
    stepsLabel.grid(row=6, column=0, padx=5, pady=5)
    stepsButton.grid(row=6, column=1, padx=5, pady=5)
    stepsEntry.grid(row=7, column=0, columnspan=2, padx=5, pady=5)
    initializeButton.grid(row=8, column=0, padx=5, pady=20)
    generateButton.grid(row=8, column=1, padx=5, pady=20)
    notifyLabel.grid(row=9, column=0, columnspan=2, padx=5, pady=5)

    # 在右侧Frame放置画布
    canvasImage = PhotoImage(file = 'ui/ui-welcome.png')
    canvasLabel = ttkbootstrap.Label(rightFrame, image=canvasImage)
    canvasLabel.grid(row=0, column=0, padx=4, pady=4)

    # 在下方Frame放置时间, xpu
    platformList = [('INTEL ', 0), ('NVIDIA', 1)]
    vPlatform = IntVar()
    if PLATFORM == "I":
        vPlatform.set(0)
    elif PLATFORM == "N":
        vPlatform.set(1)
    
    for platform, num in platformList:
        platformRadiobutton = ttkbootstrap.Radiobutton(bottomFrame, text=platform, variable=vPlatform, value=num, width=10)
        platformRadiobutton.grid(row=0, column=num, padx=5, pady=0)    
    
    if BETA_MODE:
        xpuList = [('CPU   ', 0), ('GPU 0 ', 1), ('GPU 1 ', 2)]
        vXpu = IntVar()
        vXpu.set(1)
        
        for xpu, num in xpuList:
            xpuRadiobutton = ttkbootstrap.Radiobutton(bottomFrame, text=xpu, variable=vXpu, value=num, width=10)
            xpuRadiobutton.grid(row=1, column=num, padx=5, pady=0)
    else:
        xpuList = [('CPU   ', 0), ('GPU   ', 1)]
        vXpu = IntVar()
        vXpu.set(1)
        
        for xpu, num in xpuList:
            xpuRadiobutton = ttkbootstrap.Radiobutton(bottomFrame, text=xpu, variable=vXpu, value=num, width=10)
            xpuRadiobutton.grid(row=1, column=num, padx=5, pady=0)
        
    timeLabel = ttkbootstrap.Label(bottomFrame, text='Time:                     ', width=30)
    generateProgressbar = ttkbootstrap.Progressbar(bottomFrame, length=250, style='secondary.Striped.Horizontal.TProgressbar')
    generateAllProgressbar = ttkbootstrap.Progressbar(bottomFrame, length=250, style='secondary.Striped.Horizontal.TProgressbar')
    
    timeLabel.grid(row=0, column=3, rowspan=2, padx=5, pady=5)
    generateProgressbar.grid(row=0, column=4, padx=5, pady=5)
    generateAllProgressbar.grid(row=1, column=4, padx=5, pady=5)
    
    if BETA_MODE:
        root.after(100, asyncLoopGenerate)
    
    # 进入消息循环
    root.mainloop()                 
