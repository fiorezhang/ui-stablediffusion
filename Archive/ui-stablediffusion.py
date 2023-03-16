# -*- coding: UTF-8 -*-
 
from tkinter import * 
from ttkbootstrap import Style
import time
import random

# some prompts for random generation
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

# You known you're on which platform, may change to auto-adapt in the future
PLATFORM = "I"

if PLATFORM == "I":
    from arc_ov import downloadModel, compileModel, generateImage
elif PLATFORM == "N":
    from cuda_py import downloadModel, compileModel, generateImage

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
    #negativeText.delete('1.0', END)
    negativeText.insert('1.0', random.choices(negativeList)[0]+',')
   
def seedCallback():
    seedEntry.delete(0, END)
    seedEntry.insert(0, random.randint(0, 999))
    
def stepsCallback():
    stepsEntry.delete(0, END)
    stepsEntry.insert(0, random.randint(20, 50))

def generateCallback():
    startTime = time.time()
    
    prompt = promptText.get('1.0', END).replace('\n', '')
    negative = negativeText.get('1.0', END).replace('\n', '')
    seed = seedEntry.get()
    steps = stepsEntry.get()
    
    global localPipe, canvasImage, canvasLabel
    try:
        xpuIndex = vXpu.get()
        if xpuIndex == 0: #CPU
            image = generateImage('CPU', localPipe, prompt, negative, seed, steps)
        elif xpuIndex == 1: #GPU
            image = generateImage('GPU', localPipe, prompt, negative, seed, steps)
        canvasImage = PhotoImage(file=image)
        canvasLabel.configure(image=canvasImage)
        
        endTime = time.time()
        timeLabel.configure(text='Time: ' + "%.2f"%(endTime-startTime) + 's (' + "%.2f"%(int(steps)/(endTime-startTime)) + 'it/s)')        
    except NameError:
        canvasImage = PhotoImage(file='ui/ui-initialize.png')
        canvasLabel.configure(image=canvasImage)
    except ValueError:
        canvasImage = PhotoImage(file='ui/ui-input.png')
        canvasLabel.configure(image=canvasImage)

# ### MAIN
#   
if __name__ == "__main__":    
    style = Style(theme='superhero')
    root = style.master
    root.geometry('800x600')
    root.resizable(0, 0)
    root.title('Stable Diffusion Demo v1.1')

    # 创建多个Frame
    leftFrame = Frame(root, width=250, height=520, bd=0)
    rightFrame = Frame(root, width=530, height=520, bd=0)
    bottomFrame = Frame(root, width=790, height=50, bd=0)

    leftFrame.grid(row=0, column=0, padx=5, pady=5)
    leftFrame.grid_propagate(0)
    rightFrame.grid(row=0, column=1, padx=5, pady=5)
    rightFrame.grid_propagate(0)
    bottomFrame.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
    bottomFrame.grid_propagate(0)

    # 在左侧Frame放置用户输入
    promptLabel = Label(leftFrame, text='Prompt')
    promptButton = Button(leftFrame, text='🎲', command=promptCallback)
    promptText = Text(leftFrame, width=32, height=5)    
    negativeLabel = Label(leftFrame, text='Negative Prompt')
    negativeButton = Button(leftFrame, text='🎲', command=negativeCallback)
    negativeText = Text(leftFrame, width=32, height=3)
    seedLabel = Label(leftFrame, text='Seed')
    seedButton = Button(leftFrame, text='🎲', command=seedCallback)
    seedEntry = Entry(leftFrame, width=32)
    stepsLabel = Label(leftFrame, text='Steps')
    stepsButton = Button(leftFrame, text='🎲', command=stepsCallback)
    stepsEntry = Entry(leftFrame, width=32)
    initializeButton = Button(leftFrame, text='Initialize', command=initializeCallback)
    generateButton = Button(leftFrame, text='Generate', command=generateCallback)
    notifyLabel = Label(leftFrame, text='Pls re-Initialize once switch CPU/GPU', font=('helvetica', 8))

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
    initializeButton.grid(row=8, column=0, padx=5, pady=25)
    generateButton.grid(row=8, column=1, padx=5, pady=25)
    notifyLabel.grid(row=9, column=0, columnspan=2, padx=5, pady=5)

    # 在右侧Frame放置画布
    canvasImage = PhotoImage(file = 'ui/ui-welcome.png')
    canvasLabel = Label(rightFrame, image=canvasImage)
    canvasLabel.grid(row=0, column=0, padx=4, pady=4)

    # 在下方Frame放置时间, xpu
    platformList = [('INTEL ', 0), ('NVIDIA', 1)]
    vPlatform = IntVar()
    if PLATFORM == "I":
        vPlatform.set(0)
    elif PLATFORM == "N":
        vPlatform.set(1)
    
    for platform, num in platformList:
        platformRadiobutton = Radiobutton(bottomFrame, text=platform, variable=vPlatform, value=num)
        platformRadiobutton.grid(row=0, column=num, padx=5, pady=0)    
    
    xpuList = [('CPU   ', 0), ('GPU   ', 1)]
    vXpu = IntVar()
    vXpu.set(1)
    
    for xpu, num in xpuList:
        xpuRadiobutton = Radiobutton(bottomFrame, text=xpu, variable=vXpu, value=num)
        xpuRadiobutton.grid(row=1, column=num, padx=5, pady=0)
        
    bottompad1Frame = Frame(bottomFrame, width=100, height=30, bd=0)
    timeLabel = Label(bottomFrame, text='Time: ')
    
    bottompad1Frame.grid(row=0, column=2, rowspan=2, padx=5, pady=0)
    timeLabel.grid(row=0, column=3, rowspan=2, padx=5, pady=0)
    
    # 进入消息循环
    root.mainloop()                 
