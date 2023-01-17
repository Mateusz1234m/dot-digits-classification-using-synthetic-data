from PIL import Image, ImageOps, ImageFilter, ImageStat
import numpy as np
import random
import time
import sys

# Constants
FONTS_PATH = 'fonts/'
BACKGROUNDS_PATH = 'backgrounds/'
N_DIGITS = 10
N_FONTS = 15
N_BACKGROUNDS = 19


class DigitsGenerator:
    """
        Class for synthetic data generation of digits on tyres.
    """

    def __init__(self, fonts_path, backgrounds_path,
                 digit_height_ratio_min, digit_height_ratio_max, digit_rotation_angle_min, digit_rotation_angle_max,
                 background_rotation_angle_min, background_rotation_angle_max, digit_position_max_shift,
                 noise_factor_max, resize_factor_max, digit_shift_max_x, digit_shift_max_y, color_shift_max,
                 final_resize_factor_max, shadow_blur_max, shadow_cover_ratio_min, shadow_cover_ratio_max,
                 shadow_distance_max, glow_cover_ratio_min, glow_cover_ratio_max, glow_blur_max, glow_distance_max,
                 digit_blur_max, background_size_min, background_size_max):
        """
        Initializes DigitsGenerator objects.
            :param fonts_path: path, where images of fonts are stored
            :param backgrounds_path: path, where images of backgrounds are stored
            :param digit_height_ratio_min: the minimum value of the ratio by which the image of the digit will be
                compressed vertically
            :param digit_height_ratio_max: the maximum value of the ratio by which the image of the digit will be
                compressed vertically
            :param digit_rotation_angle_min: the minimum value of the angle in degrees by which the image of the digit
                will be rotated
            :param digit_rotation_angle_max: the maximum value of the angle in degrees by which the image of the digit
                will be rotated
            :param background_rotation_angle_min: the minimum value of the angle in degrees by which the image of the
                background will be rotated
            :param background_rotation_angle_max: the maximum value of the angle in degrees by which the image of the
                background will be rotated
            :param digit_position_max_shift: the maximum value in pixels by which the image of the digit will be shifted
                on the background
            :param noise_factor_max: the maximum amplitude of noise that will be applied to the image
            :param resize_factor_max: the value of the factor by which the image of the digit will be scaled down and
                then scaled back for quality degradation
            :param digit_shift_max_x: the maximum value in pixels of the horizontal margin
            :param digit_shift_max_y: the maximum value in pixels of th vertical margin
            :param color_shift_max: the maximum color value by which the image will be shifted (brightened or darken)
            :param final_resize_factor_max: the value of the factor by which the final image of the digit will be scaled
                down andthen scaled back for quality degradation
            :param shadow_blur_max: the maximum radius of gaussian blur that will be apllied to shadow
            :param shadow_cover_ratio_min: the minimum value of parameter that specifies visibility of shadow
            :param shadow_cover_ratio_max: the maximum value of parameter that specifies visibility of shadow
            :param shadow_distance_max: the maximum distance between digit and shadow
            :param glow_cover_ratio_min: the minimum value of parameter that specifies visibility of glow
            :param glow_cover_ratio_max: the maximum value of parameter that specifies visibility of glow
            :param glow_blur_max: the maximum radius of gaussian blur that will be apllied to glow
            :param glow_distance_max: the maximum distance between glow and shadow
            :param digit_blur_max: the maximum radius of gaussian blur that will be apllied to digit
            :param background_size_min: the minimum size in pixels to which the background will be scaled
            :param background_size_max: the maximum size in pixels to which the background will be scaled
        """

        self.fonts_path = fonts_path
        self.backgrounds_path = backgrounds_path
        self.digit_height_ratio_min = digit_height_ratio_min
        self.digit_height_ratio_max = digit_height_ratio_max
        self.digit_rotation_angle_min = digit_rotation_angle_min
        self.digit_rotation_angle_max = digit_rotation_angle_max
        self.background_rotation_angle_min = background_rotation_angle_min
        self.background_rotation_angle_max = background_rotation_angle_max
        self.digit_position_max_shift = digit_position_max_shift
        self.noise_factor_max = noise_factor_max
        self.resize_factor_max = resize_factor_max
        self.digit_margin_max_x = digit_shift_max_x
        self.digit_margin_max_y = digit_shift_max_y
        self.color_shift_max = color_shift_max
        self.final_resize_factor_max = final_resize_factor_max
        self.shadow_blur_max = shadow_blur_max
        self.shadow_cover_ratio_min = shadow_cover_ratio_min
        self.shadow_cover_ratio_max = shadow_cover_ratio_max
        self.shadow_distance_max = shadow_distance_max
        self.glow_cover_ratio_min = glow_cover_ratio_min
        self.glow_cover_ratio_max = glow_cover_ratio_max
        self.glow_blur_max = glow_blur_max
        self.glow_distance_max = glow_distance_max
        self.digit_blur_max = digit_blur_max
        self.background_size_min = background_size_min
        self.background_size_max = background_size_max

    def generate_image(self, digit=None, font=None, background=None, digit_position=None,
                       digit_rotation_angle=None, background_size=None,
                       digit_height_ratio=None, background_rotation_angle=None,
                       shadow_blur=None, shadow_cover_ratio=None, shadow_angle=None, shadow_distance=None,
                       glow_blur=None, glow_cover_ratio=None, glow_distance=None, color=None,
                       digit_blur=None):
        """
            :param digit: digit that will be generated
            :param font: font in which the digit will be generated
            :param background: background on which the digit will be generated
            :param digit_position: position (tuple) of the digit relative to the center of the background
            :param digit_rotation_angle: the value of the angle in degrees by which the image of the digit will be
                rotated
            :param background_size: the size in pixels (tuple) to which the background will be scaled
            :param digit_height_ratio: the value of the ratio by which the image of the digit will be compressed
                vertically
            :param background_rotation_angle: the value of the angle in degrees by which the image of the
                background will be rotated
            :param shadow_blur: the radius of gaussian blur that will be applied to shadow
            :param shadow_cover_ratio: the value of parameter that specifies visibility of shadow
            :param shadow_angle: the angle that specifies the direction of the shadow
            :param shadow_distance: the distance between digit and shadow
            :param glow_blur: the radius of gaussian blur that will be applied to glow
            :param glow_cover_ratio: the value of parameter that specifies visibility of glow
            :param glow_distance: the distance between glow and shadow
            :param color: the color of the digit
            :param digit_blur: the radius of gaussian blur that will be applied to digit
            :return: PIL image, label
        """

        # Use the specified digit if given, otherwise draw
        digit = digit if digit is not None else str(random.randint(0, N_DIGITS - 1))
        ret_label = digit

        # Use the specified font if given, otherwise draw
        font = font if font is not None else str(random.randint(0, N_FONTS - 1))

        # Use the specified background if given, otherwise draw
        background = background if background is not None else str(random.randint(0, N_BACKGROUNDS - 1))

        # Load digit image
        digit_path = f"{self.fonts_path}/{font}/{ret_label.zfill(2)}.png"
        digit = ImageOps.grayscale(Image.open(digit_path))

        # Load background image
        background_path = f"{self.backgrounds_path}/{str(background).zfill(2)}.png"
        background = ImageOps.grayscale(Image.open(background_path))

        # Use the specified digit position if given, otherwise draw
        digit_position = digit_position if digit_position is not None else (
            random.randint(-self.digit_position_max_shift, self.digit_position_max_shift),
            random.randint(-self.digit_position_max_shift, self.digit_position_max_shift))

        # Use the specified digit rotation angle if given, otherwise draw
        digit_rotation_angle = digit_rotation_angle if digit_rotation_angle is not None else random.randint(
            self.digit_rotation_angle_min, self.digit_rotation_angle_max)

        # Use the specified digit height ratio if given, otherwise draw
        digit_height_ratio = digit_height_ratio if digit_height_ratio is not None else random.randint(
            self.digit_height_ratio_min * 100, self.digit_height_ratio_max * 100) / 100

        # Use the specified background rotation angle if given, otherwise draw
        background_rotation_angle = background_rotation_angle if background_rotation_angle is not None else \
            random.randint(self.background_rotation_angle_min, self.background_rotation_angle_max)

        # Use the specified shadow blur if given, otherwise draw
        shadow_blur = shadow_blur if shadow_blur is not None else random.randint(1, self.shadow_blur_max)

        # Use the specified shadow cover ratio if given, otherwise draw
        shadow_cover_ratio = shadow_cover_ratio if shadow_cover_ratio is not None else random.randint(
            self.shadow_cover_ratio_min, self.shadow_cover_ratio_max) / 100

        # Use the specified shadow angle if given, otherwise draw
        shadow_angle = shadow_angle if shadow_angle is not None else random.randint(0, 360)

        # Use the specified shadow distance if given, otherwise draw
        shadow_distance = shadow_distance if shadow_distance is not None else random.randint(1,
                                                                                             self.shadow_distance_max)

        # Use the specified glow blur if given, otherwise draw
        glow_blur = glow_blur if glow_blur is not None else random.randint(0, self.glow_blur_max)

        # Use the specified glow cover ratio if given, otherwise draw
        glow_cover_ratio = glow_cover_ratio if glow_cover_ratio is not None else random.randint(
            self.glow_cover_ratio_min, self.glow_cover_ratio_max) / 100

        # Use the specified glow distance if given, otherwise draw
        glow_distance = glow_distance if glow_distance is not None else random.randint(0, self.glow_distance_max)

        # Use the specified color if given, otherwise draw
        color = color if color is not None else ImageStat.Stat(background).mean[0] + random.randint(
            -self.color_shift_max, self.color_shift_max)

        # Use the specified digit blur if given, otherwise draw
        digit_blur = digit_blur if digit_blur is not None else random.randint(1, self.digit_blur_max)

        # Draw noise factor
        noise_factor = random.randint(0, self.noise_factor_max)

        # Use the specified background size if given, otherwise draw
        background_size = background_size if background_size is not None else (
            random.randint(self.background_size_min, self.background_size_max),
            random.randint(self.background_size_min, self.background_size_max))

        # Rotate digit image
        digit = digit.rotate(digit_rotation_angle, expand=1)

        # Find bounding box of the digit, crop and compress vertically
        digit = np.asarray(digit) / 255.0
        nonzero = np.where(digit > 0)
        top = min(nonzero[0])
        bottom = max(nonzero[0]) + 1
        left = min(nonzero[1])
        right = max(nonzero[1]) + 1
        digit = Image.fromarray(np.uint8(digit * 255))
        digit = digit.crop((left, top, right, bottom))
        digit = digit.resize((digit.size[0], int(digit.size[1] * digit_height_ratio)))

        # Resize and rotate background image
        background = background.resize(background_size)
        background = background.rotate(background_rotation_angle, expand=1)

        # Create separate images
        shadow = Image.new(background.mode, background.size, 0)
        glow = Image.new(background.mode, background.size, 255)
        color = Image.new(background.mode, background.size, int(color))

        # Apply noise to color
        noise = np.random.normal(0, noise_factor, size=(background.size[1], background.size[0]))
        color = Image.fromarray(np.uint8(np.clip(np.asarray(color) + noise, 0, 255)))

        # Random resize to degrade quality
        resize_factor = random.randint(100, self.resize_factor_max) / 100
        color = color.resize((int(color.size[0] / resize_factor), int(color.size[1] / resize_factor)),
                             resample=0).resize(background.size, resample=0)

        # Create alpha channels as new images
        alpha_shadow = Image.new("L", background.size, 0)
        alpha_glow = Image.new("L", background.size, 0)
        alpha_digit = Image.new("L", background.size, 0)

        # Paste digit into digit alpha channel and blur
        x_digit = int(background.size[0] / 2 + digit_position[0])
        y_digit = int(background.size[1] / 2 + digit_position[1])
        alpha_digit.paste(digit, (x_digit, y_digit))
        alpha_digit = alpha_digit.filter(ImageFilter.GaussianBlur(radius=digit_blur))

        # Paste digit into shadow alpha channel and blur
        x_shadow_shift = shadow_distance * np.sin((np.pi * shadow_angle) / 180.0)
        y_shadow_shift = shadow_distance * np.cos((np.pi * shadow_angle) / 180.0)
        x_shadow = int(x_digit + x_shadow_shift)
        y_shadow = int(y_digit + y_shadow_shift)
        digit_temp = Image.fromarray(np.uint8(np.asarray(digit) * shadow_cover_ratio))
        alpha_shadow.paste(digit_temp, (x_shadow, y_shadow))
        alpha_shadow = alpha_shadow.filter(ImageFilter.GaussianBlur(radius=shadow_blur))

        # Paste digit into glow alpha channel and blur
        x_glow_shift = glow_distance * np.sin((np.pi * (shadow_angle - 180)) / 180.0)
        y_glow_shift = glow_distance * np.cos((np.pi * (shadow_angle - 180)) / 180.0)
        x_glow = int(x_digit + x_glow_shift)
        y_glow = int(y_digit + y_glow_shift)
        digit_temp = Image.fromarray(np.uint8(np.asarray(digit) * glow_cover_ratio))
        alpha_glow.paste(digit_temp, (x_glow, y_glow))
        alpha_glow = alpha_glow.filter(ImageFilter.GaussianBlur(radius=glow_blur))

        # Apply noise to background image
        noise = np.random.normal(0, noise_factor, size=(background.size[1], background.size[0]))
        background = Image.fromarray(np.clip(np.uint8(np.asarray(background) + noise), 0, 255))

        # Blend images to final one
        ret_img = Image.composite(glow, background, alpha_glow)
        ret_img = Image.composite(shadow, ret_img, alpha_shadow)
        ret_img = Image.composite(color, ret_img, alpha_digit)

        # Crop final image
        left = x_digit - random.randint(0, self.digit_margin_max_x)
        top = y_digit - random.randint(0, self.digit_margin_max_y)
        right = x_digit + digit.size[0] + random.randint(0, self.digit_margin_max_x)
        bottom = y_digit + digit.size[1] + random.randint(0, self.digit_margin_max_y)
        ret_img = ret_img.crop((left, top, right, bottom))

        # Apply noise to final image
        ret_img = Image.fromarray(np.uint8(np.clip(
            np.asarray(ret_img) + np.random.randint(-self.color_shift_max, self.color_shift_max, size=(ret_img.size[1],
                                                                                                       ret_img.size[
                                                                                                           0])),
            0, 255)))

        # Random scale down and then up to downgrade the quality
        original_size = ret_img.size
        resize_factor = int(random.randint(100, self.final_resize_factor_max) / 100)
        ret_img = ret_img.resize((int(original_size[0] / resize_factor), int(original_size[1] / resize_factor)),
                                 resample=0).resize(original_size,
                                                    resample=0)

        return ret_img, ret_label


# Test digits generator
if __name__ == "__main__":

    # Initialize digits generator
    digits_generator = DigitsGenerator(fonts_path=FONTS_PATH, backgrounds_path=BACKGROUNDS_PATH,
                                       digit_height_ratio_min=0.3,
                                       digit_height_ratio_max=1, digit_rotation_angle_min=-45,
                                       digit_rotation_angle_max=45, background_rotation_angle_min=-30,
                                       background_rotation_angle_max=30, digit_position_max_shift=300,
                                       noise_factor_max=0, resize_factor_max=200,
                                       digit_shift_max_x=10, digit_shift_max_y=75, color_shift_max=10,
                                       final_resize_factor_max=500, shadow_blur_max=10, shadow_cover_ratio_min=50,
                                       shadow_cover_ratio_max=100,
                                       shadow_distance_max=15, glow_cover_ratio_min=0, glow_cover_ratio_max=100,
                                       glow_blur_max=1, glow_distance_max=3, digit_blur_max=4, background_size_min=600,
                                       background_size_max=1000)

    # Generate 20 random digits
    for i in range(100):
        time_start = time.time()
        img, label = digits_generator.generate_image(digit='5', font='8', background='8')
        print(f"Generation time: {time.time() - time_start}")
        print(label)
        print()
        img.save(f"temp/{i}.png")
