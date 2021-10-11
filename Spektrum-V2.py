import os
import colorsys
import PIL.ImageColor
import PIL.Image
import numpy as np
import PIL
import matplotlib.pyplot as plt



def wavelength_to_rgb(wavelength, gamma=0.8):
    '''This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R), int(G), int(B))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class Data:
    def __init__(self, image: PIL.Image.Image):
        self.image = image
        self.image_size = image.size
        self.Wavelenght = {}
        self.get_wavelenght()
        self.grey = []

    def get_grey(self):
        im = self.image.copy().convert("L")
        for x in range(self.image_size[0]):
            tempgrey = []
            for y in range(self.image_size[1]):
                tempgrey.append(im.getpixel((x, y)))
            avg = sum(tempgrey) / len(tempgrey).__round__(4)
            self.grey.append((avg))

    def get_wavelenght(self):
        for w in range(380, 750):
            r, g, b = wavelength_to_rgb(w)
            hsv = colorsys.rgb_to_hsv(r, g, b)
            self.Wavelenght[hsv[0].__round__(3)] = w


def plot(start, end, step,filename):
    greylist = np.array(D.grey)
    scale = []
    for i in range(D.image_size[0]):
        scale.append(start + i * step)
    scale = np.array(scale)
    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    im = plt.imread("temp.jpg")
    fig, ax = plt.subplots()
    im = ax.imshow(im)
    ax.plot(scale, greylist, linewidth=1, color='blue')
    ax.set_xlim([start, end])
    plt.gca().invert_yaxis()
    try:
        os.remove(f"Plots/{filename[:-4]}_plot.png")
    except Exception:
        pass
    plt.savefig(f"Plots/{filename[:-4]}_plot.png")

    # plt.show()


def convert_image(start, dif):
    mx = int(max(D.grey) + 40)
    img = D.image.copy().resize((D.image_size[0], mx))

    for x in range(D.image_size[0]):

        for y in range(mx):
            if not y >= mx - D.grey[x]-1:
                img.putpixel((x, mx - y -1), (255, 255, 255))

    img = img.resize((abs(int(dif)), mx))
    img.copy()

    white = PIL.Image.new("RGB", (int(start.__round__(0))+1, mx), "white")

    images = [white, img]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = PIL.Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save('temp.jpg')


def get_max_points(err=False):
    max_points = []
    avg = sum(D.grey) / len(D.grey)
    if not err:
        avg += avg / 2
    print("avg: ",avg)
    for g in range(len(D.grey)):
        if D.grey[g] > avg:
            max_points.append(g)

    return max_points


def get_wave_length():
    mp = get_max_points()
    starts = []
    ends = []
    step_sizes = []
    while len(mp) > 0:
        Wavelength = [(0, 0), (0, 0)]
        try:
            while abs(Wavelength[0][0] - Wavelength[1][0]) < 50:
                Wavelength = []
                for p in mp:
                    avghue = 0
                    for y in range(D.image_size[1]):
                        r, g, b = D.image.getpixel((p, y))
                        hvs = colorsys.rgb_to_hsv(r, g, b)
                        avghue += hvs[0]
                    avghue /= D.image_size[1]

                    avghue = avghue.__round__(3)
                    try:
                        Wavelength.append((p, D.Wavelenght[avghue]))
                    except Exception:
                        pass
                    mp.remove(p)
                    if len(Wavelength) == 2:
                        break
        except Exception:
            break



        wavedif = abs(Wavelength[0][1] - Wavelength[1][1])
        posdif = abs(Wavelength[0][0] - Wavelength[1][0])
        step_size = wavedif / posdif  # p = stepsize* (p-start)
        p1 = Wavelength[0][0]

        start = p1 - step_size * p1
        end = start + step_size * D.image_size[0]
        starts.append(start)
        ends.append(end)
        step_sizes.append(step_size)

    if len(starts) > 0 and len(ends) > 0:
        start = int(sum(starts) / len(starts))
        end = int(sum(ends) / len(ends))
        step_size = sum(step_sizes) / len(step_sizes)
        return start, end, step_size
    else:
        Wavelength = []
        mp = get_max_points()
        print(mp)
        for p in mp:
            avghue = 0
            for y in range(D.image_size[1]):
                r, g, b = D.image.getpixel((p, y))
                hvs = colorsys.rgb_to_hsv(r, g, b)
                avghue += hvs[0]
            avghue /= D.image_size[1]

            avghue = avghue.__round__(3)
            try:
                Wavelength.append((p, D.Wavelenght[avghue]))
            except Exception:
                pass
            mp.remove(p)
            if len(Wavelength) == 2:
                break

        wavedif = abs(Wavelength[0][1] - Wavelength[1][1])
        posdif = abs(Wavelength[0][0] - Wavelength[1][0])
        step_size = wavedif / posdif  # p = stepsize* (p-start)
        p1 = Wavelength[0][0]
        print("Wavelenght: ",Wavelength)
        start = p1 - step_size * p1
        end = start + step_size * D.image_size[0]
        if end > 700:
            end = 700
        if start > 800 or start < 400:
            start = 400

        return start, end, step_size




if __name__ == '__main__':
    for f in os.listdir("Spektren"):
        img = PIL.Image.open(f"Spektren/{f}")
        D = Data(img)
        D.image_size = (int(D.image_size[0] / 1), int(D.image_size[1] / 1))
        D.image.resize(list(D.image_size))
        D.get_grey()
        start, end, step = get_wave_length()
        print(start, end)
        convert_image(start, end - start)
        plot(start, end, step,f)
        print(f)
    plt.show()
