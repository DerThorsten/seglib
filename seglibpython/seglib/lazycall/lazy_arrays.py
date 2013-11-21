import os
from vigra import readImage
import h5py
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
     FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer
from colorama import Fore, Back, Style


import os.path
#from termcolor import colored


def getFiles(path,ending):
    fnames = []
    baseNames = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(ending):
                fp          =  os.path.join(root, f)
                baseName    =  f.split('.')[0]
                #print baseName
                fnames.append(fp)
                baseNames.append(baseName)
    return fnames,baseNames



def makeFullPath(folder,baseNames,ending):
    fps = []
    if folder[-1]=='/':
        f=folder
    else :
        f='%s/'%folder

    if ending[0]=='.':
        e=ending
    else :
        e='.%s'%ending


    for baseName in baseNames :
        fp="%s%s%s"%(f,baseName,e)
        #print fp
        fps.append(fp)

    return fps


def pBar(size,name=""):
    #print(Back.CYAN+Fore.BLACK + self.name + Fore.RESET + Back.RESET + Style.RESET_ALL)
    widgets = [ Back.CYAN+Fore.BLACK ," %s :"%name ,Percentage(), ' ', 
        Bar(marker=RotatingMarker()), ' ', ETA(), ' ', FileTransferSpeed(),Back.RESET + Style.RESET_ALL]

    if size == 1 :
        size+=1
    pbar = ProgressBar(widgets=widgets, maxval=size-1).start()
    return pbar




def h5Exist(f,dset):
    try :
        if os.path.isfile(f):
            h5file = h5py.File(f,'r')
            e = False
            if dset in h5file.keys():
                #h5file[dset]
                e = True
            h5file.close()

            return e
        return False
    except:
        return False



class LazyArrays(object):
    def __init__(self,files,dset=None,filetype="h5"):
        self.files      = files 
        self.dset       = dset
        self.filetype   = filetype

    def __len__(self):
        return len(self.files)


    def __getitem__(self,index):

        if self.filetype == "image" :
            return readImage(self.files[index])
        elif self.filetype == "h5" :
            f = h5py.File(self.files[index],'r')
            value = f[self.dset].value
            f.close()
            return value






class LazyCaller(object):
    def __init__(self,f,skip=False,verbose=True,overwrite=False,
                name="",compress=False,skipAll=False,compressionOpts=2):
        self.f = f
        self._batchKWargs   = set()
        self.verbose        = verbose
        self.outputFiles    = None
        self.dset           = None
        self.overwrite      = overwrite
        self.name           = name
        self.compress       = compress
        self.compressionOpts= compressionOpts
        self.skipAll        = skipAll

    def setCompression(self,compress,compressionOpts=2):
        self.compress=compress
        self.compressionOpts=compressionOpts

    def setBatchKwargs(self,batchKWargs):
        self._batchKWargs = set(batchKWargs)

    def setOutput(self,files,dset):
        self.outputFiles=files
        self.dset=dset

    def __call__(self,*args,**kwargs):

        #print colored('Compute ', 'red'), colored(self.name, 'green')

        #from colorama import Fore, Back, Style
        #print(Back.CYAN+Fore.BLACK + self.name + Fore.RESET + Back.RESET + Style.RESET_ALL)
        #print( Back.GREEN+ 'some red text')
        #print(Back.GREEN + 'and with a green background')
        #print(Style.DIM + 'and in dim text')
        #print(Fore.RESET + Back.RESET + Style.RESET_ALL)
        #print('back to normal now')

        assert self.outputFiles is not None        
        if len(args)>0 :
            raise RuntimeError("LazyCaller(...) does only support keyword arguments")
        if self.skipAll :
            if self.verbose :
                print(Back.CYAN+Fore.BLACK +" SKIP %s"%self.name+Back.RESET + Style.RESET_ALL)
        else :
            constKwargs=dict()
            batchKwargs=dict()
            callingKwargs=dict()   
            batchLen = None 
            for kwarg in kwargs.keys() :
                if kwarg in self._batchKWargs :
                    batchInput          = kwargs[kwarg]
                    batchKwargs[kwarg]  = batchInput

                    if batchLen is None :
                        batchLen = len(batchInput)
                    else:
                        assert batchLen == len(batchInput) 
                else :
                    constKwargs[kwarg]=kwargs[kwarg]
                    callingKwargs[kwarg]=kwargs[kwarg]


            if self.verbose :
                pbar = pBar(batchLen,name=self.name)
            # iterate over all batch items 



            for batchIndex in range(batchLen):


                #check if we need to do the computation
                exist = h5Exist(self.outputFiles[batchIndex],self.dset)
                if exist == False  or self.overwrite == True:

                    # set up the kwargs for a single function call
                    for batchKwarg in self._batchKWargs :
                        batchItem  = kwargs[batchKwarg][batchIndex]
                        callingKwargs[batchKwarg]=batchItem

                    #print "batchIndex",batchIndex,"len",batchLen
                    # call the actual function and store the result
                    self._local_call_(callingKwargs,self.outputFiles[batchIndex])

                if self.verbose :
                    pbar.update(batchIndex)
            if self.verbose :
                pbar.finish()


    def _local_call_(self,callingKwargs,outputFile):

        result = self.f(**callingKwargs)

        f = h5py.File(outputFile,'w')

        if self.compress :
            dataset = f.create_dataset(self.dset,shape=result.shape,compression='gzip',compression_opts=self.compressionOpts) 
            dataset[...]=result
        else :
            f[self.dset]=result

        
        #f[self.dset] =result
        f.close()



if __name__ == "__main__" :


    imagePath   = "/home/tbeier/src/privatOpengm/experiments/datasets/bsd500/BSR/BSDS500/data/images/test/"
    files       = getFiles(imagePath,"jpg")
    images      = LazyArrays(files,filetype="image") 

    print type(images)

    img = images[0]


    for img in images :
        print img.shape