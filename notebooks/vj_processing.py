from k2utils import *
import sys

def running_func(func,x,N, **func_kwargs):
    
    n,m = N/2, N%2
    return np.array([func(x[i-n if i > n else i:i+n+m], **func_kwargs) for i in xrange(len(x))])

runningMean = partial(running_func,np.nanmedian)
#def runningMean(x,N):
#    return np.array([np.nanmedian(x[i-N/2 if i > N/2 else i:i+N/2+(N%2)]) for i in xrange(len(x))])

def runningStd(x,N):
    return np.array([np.nanstd(x[i-N/2 if i > N/2 else 0:i+N/2 +(N%2)]) for i in xrange(len(x))])



def plotNiceIntervals(ax,x,y,N=30, label=None, color=None):
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    rm = runningMean(y,N)
    #rs = runningStd(y,N)
    #rs = runningMean(rs,N) #Make it nice and smooth
    
    ax.plot(x, y, ls=' ', marker='.', ms=1, color=color)
    #ax.fill_between(x, rm+rs, rm-rs, color=color, alpha=0.1, linestyle='None')
    ax.plot(x, rm, ls='-', color=color, lw=2, label=label)
    #ax.plot(x, rm, ls='-', color=color, lw=1, marker='.', ms=1)
    #ax.set_xlim(min(x), max(x))
    #ax.set_ylim(np.nanmean(rm)-np.nanstd(rm)*3, np.nanmean(rm)+np.nanstd(rm)*3)


def axes_prop_setter(local_vars, *args, **kwargs):

    locals().update(local_vars)
    for key, value in kwargs.iteritems():
        if hasattr(value, '__iter__'):
            value = iter(value)
        elif type(value) is str:
            if value.startswith('@'): # for custom expressions
                value = eval(value[1:])
        else:
            value = iter([value])
        eval('ax.{}(*value)'.format(key))
    for arg in args:
        eval('ax.{}()'.format(arg))
        

def get_pixel_lc(gen, percentiles, mag_range=None):
    '''percentiles is a flat array of percentiles ranging from 0 to 100'''
    global N
    higher, lower = None, None
    if mag_range is None:
        higher, lower = np.inf, -np.inf
    else:
        higher, lower = magToFlux(np.array(sorted(mag_range)))
    if np.inf in mag_range:
        lower = -np.inf
    
    ccd = gen.ccd
    M = len(percentiles)-1
    
    lc = np.empty((N,M,5)) # min, max, var, median, mean
    lc[::] = np.nan
        
    funcs = (np.var, np.median, np.mean)
    N = len(gen.containers.containers[0].pixels)
    with LoadingBar(True) as lb:    
        
        for i,g in enumerate((gen.get_unordered(i) for i in xrange(N))):
            m = (g > lower) & (g < higher)
            if m.any():
                p = np.percentile(g[m], percentiles)
                lc[i,:,:2] = zip(p[:-1], p[1:])
                
                # New
                for j, (low, high) in enumerate(lc[i,:,:2]):
                    cut = g[(g > low) & (g < high)]
                    if len(cut):
                        lc[i,j,2] = np.var(cut)
                        lc[i,j,3] = np.median(cut)
                        #lc[i,j,4] = np.dot(cut,np.ones(len(cut)))/len(cut)
                        lc[i,j,4] = np.mean(cut)
            lb.update_bar(1.0*i/N)

    return lc

def plot_pixel_lc(args, title=None, ylim=None, ax = None, *axargs, **axkwargs):
    
    if ax is None:
        fig, ax = subplots(1,1, figsize=(16,9))
    else:
        fig = ax.figure
    ax.set_prop_cycle(color=rplot.palettable.cubehelix.perceptual_rainbow_16.mpl_colors)
    #ax.set_prop_cycle(color=rplot.palettable.colorbrewer.sequential.YlGnBu_6.mpl_colors)
    for t, y, label, b in args:
        plotNiceIntervals(ax, t, y, label=label, N=120)
    
    if title is not None:
        ax.set_title(title)
        
    ax.set_xlabel('Date')
    ax.set_ylabel('Flux (Median Normalized)')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(min(t), max(t))
    legend = ax.legend()
    for line in legend.get_lines():
        line.set_alpha(1)
        line.set_marker('.')
    axes_prop_setter(locals(), *axargs, **axkwargs)
    return fig, ax

def correct_target_pixels(EPIC, lc):
    '''correct the target pixels of an object based on the percentiles input
        more percentiles means better resolution but more noise
        
        We'll do 3 corrections.  First on the raw target pixels, then on
        the VJ and EVEREST light curves
        
        lc is the pixel lc'''
    
    urls = (get_url(EPIC, 8) for get_url in (get_target_pixel_url, get_vj_url, get_everest_url))
    hdus = fits_downloader(urls)
    
    # First we'll do the hard part, target pixel correction
    
    target_pixels = BasicFitsContainer('PIX',hdus.next(),'FLUX')
    corrected_target_pixels = np.empty(target_pixels.pixels.shape)*np.nan
    #corrected_target_pixels[::] = target_pixels.pixels[::]
    for i in xrange(N):
        '''for each cadence'''
        m,n = target_pixels.pixels[i].shape
        bins = np.concatenate((lc[i,:,0], [lc[i,-1,1]]))
        for j in xrange(m):
            for k in xrange(n):
                '''for each pixel in the frame, we determine the 
                bin to which the pixel belongs in the lc
                
                then we subtract the difference between the median of the bin at this cadence
                and the median of the median of the bin over all cadenences'''
                
                pixel = target_pixels.pixels[i,j,k]
                if np.isnan(pixel) or pixel <= 0: # NaN pixel or negative
                    target_pixels.pixels[i,j,k] = np.nan
                    continue
                
                #pos = np.where((pixel >= low) & (pixel <= high))[0] # This is the corresponding correction
                try:
                    pos = np.digitize(pixel, bins)
                except IndexError: # the pixel is not NaN so there must be a corresponding entry
                    print "Error: No Good Percentile Found (This is a bug, one should always be found)"
                    print "  The pixel value was {}".format(pixel)
                    print "  While the range of bins was ({},{})".format(low[0], high[-1])
                    continue
                    
                if not pos: # negative value
                    target_pixels.pixels[i,j,k] = np.nan
                    continue
                if pos == len(bins): # too high
                    target_pixels.pixels[i,j,k] = np.nan
                    continue
                
                pos -= 1 #set to the correct index now
                
                    
                pixel_lc_median = np.nanmedian(lc[:,pos,3]) # median of medians 
                correction = lc[i,pos,3] - pixel_lc_median # the median at this position - the full median  
                corrected_pixel = pixel - correction # subtract off the correction
                
                #print pixel, lc[i,pos,:2], pixel_lc_median, lc[i,pos,3], correction, corrected_pixel
                #return
                
                if np.isnan(corrected_pixel):
                    raise ValueError("Bad Pixel Correction\n  pixel value: {}\n  pixel lc median: {}".format(pixel, pixel_lc_median))
                
                corrected_target_pixels[i,j,k] = corrected_pixel
    
    # Now we'll try to naively correct the VJ light curves
    
    hdu = hdus.next()
    flux = magToFlux(hdu[0].header['KEPMAG'])
    #flux = np.nanmedian(corrected_target_pixels)
    #print flux
    vj_pixels = BasicFitsContainer('VJ',hdu,'FRAW')
    vj_pixels.FRAW *= flux
    
    corrected_vj_pixels = np.empty(vj_pixels.FRAW.shape)*np.nan
    #corrected_target_pixels[::] = target_pixels.pixels[::]
    for i in xrange(len(vj_pixels.FRAW)):
        '''for each cadence'''
        bins = np.concatenate((lc[i,:,0], [lc[i,-1,1]]))
        '''for each pixel in the frame, we determine the 
        bin to which the pixel belongs in the lc

        then we subtract the difference between the median of the bin at this cadence
        and the median of the median of the bin over all cadenences'''

        pixel = vj_pixels.FRAW[i]
        if np.isnan(pixel) or pixel <= 0: # NaN pixel or negative
            vj_pixels.FRAW[i] = np.nan
            continue

        #pos = np.where((pixel >= low) & (pixel <= high))[0] # This is the corresponding correction
        try:
            pos = np.digitize(pixel, bins)
        except IndexError: # the pixel is not NaN so there must be a corresponding entry
            print "Error: No Good Percentile Found (This is a bug, one should always be found)"
            print "  The pixel value was {}".format(pixel)
            print "  While the range of bins was ({},{})".format(low[0], high[-1])
            continue

        if not pos: # negative value
            vj_pixels.FRAW[i] = np.nan
            continue
        if pos == len(bins): # too high
            vj_pixels.FRAW[i] = np.nan
            continue

        pos -= 1 #set to the correct index now


        pixel_lc_median = np.nanmedian(lc[:,pos,3]) # median of medians 
        correction = lc[i,pos,3] - pixel_lc_median # the median at this position - the full median  
        corrected_pixel = pixel - correction # subtract off the correction

        #print pixel, lc[i,pos,:2], pixel_lc_median, lc[i,pos,3], correction, corrected_pixel
        #return

        if np.isnan(corrected_pixel):
            raise ValueError("Bad Pixel Correction\n  pixel value: {}\n  pixel lc median: {}".format(pixel, pixel_lc_median))

        corrected_vj_pixels[i] = corrected_pixel
                
    return corrected_target_pixels, target_pixels, corrected_vj_pixels, vj_pixels
                
                
def test(EPIC, lc):
    
    ctp, tp, cvj, vj = correct_target_pixels(EPIC, lc)
    
    # for saving the corrected fits file
    urls = (get_url(EPIC, 8) for get_url in (get_target_pixel_url, get_vj_url, get_everest_url))
    hdus = fits_downloader(urls)
    hdu = hdus.next()
    hdu[1].data['FLUX'][::] = ctp
    hdu[1].data['FLUX'][np.isnan(ctp)] = 0.0
    hdu.writeto("../data/test_fits_correction.fits", clobber=True)
    
    t = 0.02043229*np.arange(N)+2559.06849083
    
    y1 = np.nanmedian(tp.pixels, axis=(1,2))
    y2 = np.nanmedian(ctp, axis=(1,2))
    y3 = cvj
    y4 = vj.FRAW
    
    #corrections = tp.pixels - ctp
    #low_idx = np.array([(d < np.nanmedian(d)) for d in tp.pixels])
    #print low_idx.shape
    #high_idx = ~low_idx
    
    #additions = np.array([np.nanmean(c.ravel()[low.ravel()]) for c, low in zip(corrections, low_idx)])[:len(vj.FRAW)]
    #subtractions = np.array([np.nanmean(c.ravel()[high.ravel()]) for c, high in zip(corrections, high_idx)])[:len(vj.FRAW)]
    
    #y3 = vj.FRAW - additions + subtractions
    
    y1 = sigma_clip(y1,5)
    y2 = sigma_clip(y2,5)
    y3 = sigma_clip(y3,5)
    y4 = sigma_clip(y4,5)
    
    
    args = ((t, y1,'Original', None), (t, y2, 'Corrected', None),(vj.t, y4, 'vj', None),(vj.t, y3, 'vj Corrected', None))
    
    plot_pixel_lc(args, title=None, ylim=None)
    
    #print y1
    #print y2
    #print x
    

    
    #fig, ax = subplots(1,1, figsize=(16,9), dpi=80)
    
    
    #plotNiceIntervals(ax, t, y1, label='original', N=120)
    #plotNiceIntervals(ax, t, y2, label='corrected', N=120)
    
    #ax.legend()
    
    
    #ax.set_ylim(0,30)
    
def sigma_clip(data, n_sigma, n_iter=120, axis=None):
    
    
    new_data = np.empty(data.shape)
    new_data[::] = data[::]
    std = np.nanstd(new_data, axis=axis)*n_sigma
    for _ in xrange(n_iter):
        new_std = std
        mean = np.nanmedian(new_data, axis=axis)
        bounds = (mean-std, mean+std)
        outliers = (new_data < bounds[0]) | (new_data > bounds[1])
        new_data[outliers] = np.nan
        std = np.nanstd(new_data, axis=axis)*n_sigma
        if axis is None:
            test = std == new_std
        else:
            test = (std == new_std).all()
        if test:
            break

    return new_data


def verify_execution(func): # verify a method has been run

    def _wrapper(self, *args, **kwargs):

        try:
            result = func(self, *args, **kwargs)
            success = True

        except:
            success = False
            raise

        else:
            return result

        finally:
            self.sanity_checks[hash(func)] = success

    return _wrapper

ve = verify_execution # for brevity

class SanityCheckError(Exception):

    def __init__(self, func, host, error):

        self.func_name = func.im_func.func_name # function we check sanity for
        self.host_name = host.im_func.func_name # function that depends on func

        self.messenger = {
            'fail':self.method_failed_message,
            'notrun':self.method_not_run_message
        }[error] # failure condition (method failed vs method not run)

        super(SanityCheckError, self).__init__(self.create_message())
        self.errors = {'fail':1,'notrun':2}[error]


    def create_message(self):

        end_message = "Required by '{0}'".format(self.host_name)
        message = '\n  '.join((self.messenger(), end_message))
        return message


    def method_failed_message(self):

        message = "Execution of method '{0}' failed".format(self.func_name)
        return message

    def method_not_run_message(self):

        message = "Method '{0}' was never executed".format(self.func_name)    
        return message

def link_sanity(host, func):
    '''host depends on func'''

    def _wrapper(self, *args, **kwargs):

        _hash = hash(func)
        if _hash not in self.sanity_checks:
            raise SanityCheckError(func, host, 'notrun')

        if not _self.sanity_checks[_hash]:
            raise SanityCheckError(func, host, 'fail')
    
        return host(self, *args, **kwargs)

    return _wrapper

class VJPipeline:


    def __init__(self, EPIC, ccd):

        self.EPIC = EPIC
        self.ccd = ccd


        self.pix = None # target pixels
        self.vj = None #Vanderberg Johnson Processed

        self.aperture = None # VJ aperture grid
        self.aperture_name = None

        self.sanity_checks = {}
        #self.set_aperture = link_sanity(self.set_aperture, self.download)

        self.index = 0

    #@ve
    def download(self):

        #urls = (get_url(self.EPIC, self.ccd.campaign) for get_url in (get_target_pixel_url, get_vj_url))
        urls = (get_url(self.EPIC, self.ccd.campaign) for get_url in (get_target_pixel_url, get_lc_url))
        hdus = fits_downloader(urls)
        self.pix, self.vj = hdus

        self.images = np.array(self.pix['TARGETTABLES'].data['FLUX'])
        #self.bkg_images = np.array(self.pix['TARGETTABLES'].data['FLUX_BKG'])

        #good_quality_mask = np.in1d(self.pix['TARGETTABLES'].data['TIME'], self.vj[1].data['T'])
        good_quality_mask = np.array([True]*len(self.images))    

        #self.images = self.images[good_quality_mask]
        #self.bkg_images = self.bkg_images[good_quality_mask]

        #self.images += self.bkg_images
        
        self.good_quality_mask = good_quality_mask

        self.photometry = np.empty(self.images.shape[0])*np.nan

    #@ve
    def set_aperture(self, aperture, index):
        '''aperture is either 'CIRC_APER_TBL' or 'PRF_APER_TBL'
            index is from 0 to 9
        '''

        self.aperture = np.array(self.pix['APERTURE'].data > 0)
        #self.aperture = np.array(self.vj[aperture].data[index], dtype=np.bool)
        #self.aperture_name = aperture[:-4] + str(index)
        self.mask = self.aperture


    def run_pipeline(self):

        print "Running Pipeline"
        n_images = self.images.shape[0]

        self.index = 0
        # extract photometry estimate
        print "Extracting Photometry"
        for self.index in xrange(n_images):
            self.start_per_image_pipeline()
            self.compute_background_flux()
            self.extract_photometry()

        # perform continuum-normalization
        print "Normalizing"
        self.photometry = np.array(self.photometry)
        self.continuum_normalize()

        # compute object position
        print "Computing Positions"
        self.index = 0
        for self.index in xrange(n_images):
            self.start_per_image_pipeline()
            self.find_centroid()            

    def start_per_image_pipeline(self):

        self.image = self.images[self.index]
        self.notnanmask = ~np.isnan(self.image)

        self.in_aper = self.image[self.mask & self.notnanmask]
        self.out_aper = self.image[(~self.mask) & self.notnanmask]

    def compute_background_flux(self):
        '''generator that yields background fluxes by each cadence'''
        
        self.background = np.median(self.out_aper)
        if np.isnan(self.background):
            self.background = 0.0
        self.background = 0
        #self.background = 0
        #assert not np.isnan(self.background), "Background is NaN"

    def extract_photometry(self):

        self.photometry[self.index] = np.sum(self.in_aper - self.background)
        #assert not np.isnan(self.photometry[self.index]), "Photometry is NaN"

    def continuum_normalize(self):
        '''performed after we have photometry for each object'''

        pass
        #self.photometry /= np.nanmedian(self.photometry)
        #assert not np.isnan(self.photometry).any(), "Median is NaN"

    def find_centroid(self):

        
        self.center_of_flux()

    def center_of_flux(self):

        image = self.image
        mask = self.notnanmask

        grids = np.meshgrid(*map(np.arange, image.shape), sparse=True)
        f_total = np.sum(image[mask])

        self.centroids = ((image*grid)[mask]/f_total for grid in grids) #yc, xc
        self.grids = grids

    def gaussian_LM(self):
        '''WARNING: Not Implimented, Can only choose cof for centroid'''
        A = self.image[int(self.yc + 0.5), int(self.xc + 0.5)]

        self.widths = self.image.shape
        z_params = zip(self.grids, self.centroids, self.widths)
        z = np.sum([((g - c)/w)**2 for g, c, w in z_params])


    def compare(self):

        fig, ax = subplots(1,1)
        #vj_t = self.vj[self.aperture_name].data['T']
        #vj_f = self.vj[self.aperture_name].data['FRAW']

        #vj_t = self.vj['LIGHTCURVE'].data['TIME']
        #vj_f = self.vj['LIGHTCURVE'].data['SAP_FLUX']
        #plotNiceIntervals(ax, vj_t, vj_f, label='Original SAP LC')

    

        #quality = self.pix[1].data['QUALITY'] # determine the quality mask later

        t = self.pix['TARGETTABLES'].data['TIME']
        f = self.photometry

        #mask = np.in1d(t, vj_t)

        #plotNiceIntervals(ax, t[mask], f, label='My Processing')
        plotNiceIntervals(ax, t, f, label='My Processing')
        #ax.set_ylim(0.5,1.75)
        #diff = f - vj_f
        #print np.sum(diff), np.mean(diff), np.median(diff), np.std(diff), len(np.where(diff != 0)[0])

        ax.legend()
        #savefig('test_processing.png')

        return ax

    def check_sanity(self, func, host):

        _hash = hash(func)
        if _hash not in self.sanity_checks:
            raise SanityCheckError(func, host, 'notrun')

        if not _self.sanity_checks[_hash]:
            raise SanityCheckError(func, host, 'fail')
        


def correct_target_pixels(EPIC, lc, aperture):
    '''correct the target pixels of an object based on the percentiles input
        more percentiles means better resolution but more noise
        
        We'll do 3 corrections.  First on the raw target pixels, then on
        the VJ and EVEREST light curves
        
        aperture is a mask'''
    
    #percentiles = np.arange(0,100.1,0.1)
    #lc = get_pixel_lc(gen, percentiles)
    hdus = fits_downloader([get_target_pixel_url(EPIC, 8)])
    
    # First we'll do the hard part, target pixel correction
    
    target_pixels = BasicFitsContainer('PIX',hdus.next(),'FLUX')
    corrections = np.empty(target_pixels.pixels.shape)*np.nan
    corrected_target_pixels = np.empty(target_pixels.pixels.shape)*np.nan
    N = len(target_pixels.pixels)
    for i in xrange(N):
        '''for each cadence'''
        m,n = target_pixels.pixels[i].shape
        bins = np.concatenate((lc[i,:,0], [lc[i,-1,1]]))
        for j in xrange(m):
            for k in xrange(n):
                '''for each pixel in the frame, we determine the 
                bin to which the pixel belongs in the lc
                
                then we subtract the difference between the median of the bin at this cadence
                and the median of the median of the bin over all cadenences'''
                
                pixel = target_pixels.pixels[i,j,k]
                if np.isnan(pixel) or pixel <= 0: # NaN pixel or negative
                    target_pixels.pixels[i,j,k] = np.nan
                    continue
                
                #pos = np.where((pixel >= low) & (pixel <= high))[0] # This is the corresponding correction
                try:
                    pos = np.digitize(pixel, bins)
                except IndexError: # the pixel is not NaN so there must be a corresponding entry
                    print "Error: No Good Percentile Found (This is a bug, one should always be found)"
                    print "  The pixel value was {}".format(pixel)
                    print "  While the range of bins was ({},{})".format(low[0], high[-1])
                    continue
                    
                if not pos: # negative value
                    print "Warning: Pixel lower than any bin, should not happen"
                    target_pixels.pixels[i,j,k] = np.nan
                    continue
                if pos == len(bins): # too high
                    print "Warning: Pixel greater than any bin, should not happen"
                    target_pixels.pixels[i,j,k] = np.nan
                    continue
                
                pos -= 1 #set to the correct index now
                
               
                # Now we're going to subtract off a model
                # made from the pixel light curve minus it's median     
                pixel_lc_median = np.nanmedian(lc[:,pos,3]) # median of medians 
                correction = lc[i,pos,3] - pixel_lc_median # the median at this position - the full median  
                corrections[i,j,k] = correction
                #corrected_pixel = pixel - correction # subtract off the correction
                
                #print pixel, lc[i,pos,:2], pixel_lc_median, lc[i,pos,3], correction, corrected_pixel
                #return
                
                if np.isnan(correction):
                    raise ValueError("Bad Pixel Correction\n  pixel value: {}\n  pixel lc median: {}".format(pixel, pixel_lc_median))
                
        corrected_target_pixels[i] = target_pixels[i,:,:]
        corrected_target_pixels[i][aperture] -= corrections[aperture]
        corrected_target_pixels[i][~aperture] += corrections[~aperture]
        
    return corrected_target_pixels, corrections 


if __name__ == '__main__':


    ccd = CCD(campaign=8, field='FLUX', channel=2, module=6)
    cont = load_pixel_map(ccd)
    gen=PixMapGenerator(cont, cache=True)


    # module 6, channel 2 {
    EPIC = 220282234
    EPIC = 220268684 # Faint object (~19th magnitude) AGN with massive trends
    EPIC = 220284690 #ditto
    EPIC = 220289355 # ditto again
    # }


    pipeline = VJPipeline(EPIC, ccd)
    pipeline.download()
    pipeline.set_aperture('CIRC_APER_TBL',9)
    pipeline.run_pipeline()

    pipeline.compare()


