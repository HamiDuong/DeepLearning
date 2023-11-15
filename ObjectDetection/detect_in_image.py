import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import pyautogui
from pywinauto.application import Application
import pytesseract
from pytesseract import Output
import time


def load_model(path_to_saved_model_folder):
    """
    Load an object detection from  the given path
    :param path_to_saved_model_folder: relative path string to the folder saved_model of the wanted model
    :return: TensorflowObject
    """
    model = tf.saved_model.load(path_to_saved_model_folder)
    return model


def int_from_label(label):
    """
    Gives the Integer value of a class label
    :param label: string - name of the class
    :return: corresponding Integer value
    """
    label_dict_label2num = {    
        'SystemBackIcon': 1,
        'HomeIcon': 2,
        'Screen': 3,
        'RadioButtonFalse': 4,
        'BlockWithRadioButton': 5,
        'BlackButton': 6,
        'Scrollbar': 7,
        'InfoIconSmall': 8,
        'TimeTile': 9,
        'BlueButton': 10,
        'xIcon': 11,
        'ArrowLeft': 12,
        'InfoIcon': 13,
        'PopUpWindow': 14,
        'DealerEntry': 15,
        'ArrowRight': 16,
        'RadioButtonTrue': 17,
        'ListRight': 18,
        'ListLeft': 19,
        'Tile': 20,
        'DateTile': 21,
        'UserInput': 22,
        'BlockWithArrow': 23,
        'BlockWithBoxSelect': 24,
        'OverviewEntry': 25,
        'PenIcon': 26,
        'SettingsWindow': 27,
        'BoxSelectFalse': 28,
        'TabBar': 29,
        'ButtonWithToggle': 30,
        'SummaryEntry': 31,
        'MenuApp': 32,
        'App': 33,
        'ToggleTrue': 34,
        'LanguageSetting': 35,
        'Keyboard': 36,
        'ToggleFalse': 37,
        'SearchIcon': 38,
        'MenuBar': 39,
        'BoxSelectTrue': 40,
        'TrashCanIcon': 41
    }
    return label_dict_label2num[label]


def label_from_int(num):
    """
    Gives the class label of an Integer value
    Input   num: int - id of the class label
    Output  string: corresponding string value
    """
    label_dict_num2label = {
        1: 'SystemBackIcon',
        2: 'HomeIcon',
        3: 'Screen',
        4: 'RadioButtonFalse',
        5: 'BlockWithRadioButton',
        6: 'BlackButton',
        7: 'Scrollbar',
        8: 'InfoIconSmall',
        9: 'TimeTile',
        10: 'BlueButton',
        11: 'xIcon',
        12: 'ArrowLeft',
        13: 'InfoIcon',
        14: 'PopUpWindow',
        15: 'DealerEntry',
        16: 'ArrowRight',
        17: 'RadioButtonTrue',
        18: 'ListRight',
        19: 'ListLeft',
        20: 'Tile',
        21: 'DateTile',
        22: 'UserInput',
        23: 'BlockWithArrow',
        24: 'BlockWithBoxSelect',
        25: 'OverviewEntry',
        26: 'PenIcon',
        27: 'SettingsWindow',
        28: 'BoxSelectFalse',
        29: 'TabBar',
        30: 'ButtonWithToggle',
        31: 'SummaryEntry',
        32: 'MenuApp',
        33: 'App',
        34: 'ToggleTrue',
        35: 'LanguageSetting',
        36: 'Keyboard',
        37: 'ToggleFalse',
        38: 'SearchIcon',
        39: 'MenuBar',
        40: 'BoxSelectTrue',
        41: 'TrashCanIcon',
    }
    return label_dict_num2label[num]


def find_center(ymin, xmin, ymax, xmax):
    """
    Calculates the center point of a bounding box
    Input   ymin: float
            xmin: float
            ymax: float
            xmax: float
    Output  keypoint_x : float
            keypoint_y : float
    """
    keypoint_x = xmin + ((xmax - xmin)/2)
    keypoint_y = ymin + ((ymax - ymin)/2)
    return keypoint_x, keypoint_y


def detect_on_image(model, img_path, threshold, output_path):
    """
    Perform object detection and detect all objects with the given model on the jpg-file
    Objects get counted if they reach the confidence threshold
    Input   model: TensorflowObject - loaded with the function load_model
            img_path: string - path to the jpg-file
            threshold: float - value between [0:1], confidence value to accept a detected object
            output_path: string - save path for the image with the bounding boxes drawn in
    """
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    detection = model(input_tensor)

    detection_boxes = detection['detection_boxes'][0]
    detection_classes = detection['detection_classes'][0]
    detection_scores = detection['detection_scores'][0]

    for box, class_id, score in zip(detection_boxes, detection_classes, detection_scores):
        if score > threshold:
            ymin, xmin, ymax, xmax = box
            ymin = int(ymin * height)
            xmin = int(xmin * width)
            ymax = int(ymax * height)
            xmax = int(xmax * width)

            draw_bounding_box(ymin, xmin, ymax, xmax, class_id, score, image, "")

            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # class_label = label_from_int(int(class_id))
            # score_percent = int(score * 100)

            # label = f'{class_label}: {score_percent}%'
            # cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            pass
    cv2.imwrite(output_path, image)


def detect_on_screenshot(model, screenshot, threshold, output_path):
    """
    Perform object detection and detect all object with the given model on a screenshot taken with pyautogui.screenshot
    Input   model: TensorflowObject - loaded with the function load_model
            screenshot: PILimage - screenshot taken with pyautogui.screenshot
            threshold: float - value between [0:1], confidence value to accept a detected object
            output_path: string - save path for the image with the bounding boxes drawn in
    """
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    height, width, _ = image.shape
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    detection = model(input_tensor)

    detection_boxes = detection['detection_boxes'][0]
    detection_classes = detection['detection_classes'][0]
    detection_scores = detection['detection_scores'][0]

    for box, class_id, score in zip(detection_boxes, detection_classes, detection_scores):
        if (score > threshold) == True:
            ymin, xmin, ymax, xmax = box
            ymin = int(ymin * height)
            xmin = int(xmin * width)
            ymax = int(ymax * height)
            xmax = int(xmax * width)

            draw_bounding_box(ymin, xmin, ymax, xmax, class_id, score, image, "")

            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # class_label = label_from_int(int(class_id))
            # score_percent = int(score * 100)

            # label = f'{class_label}: {score_percent}%'
            # cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            pass
    cv2.imwrite(output_path, image)


def draw_bounding_box(ymin, xmin, ymax, xmax, class_id, score, image, text):
    """
    Draw a rectange on an image with the label and confidence score
    Input   ymin
            xmin
            ymax
            xmax
            class_id
            score
            image
            text
    Output  None
    """
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    class_label = label_from_int(int(class_id))
    score_percent = int(score * 100)

    if text != "":
        class_label = text + " " + class_label

    label = f'{class_label}: {score_percent}%'
    cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def draw_bounding_box_around_text(x, y, w, h, image, text):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f'{text}'
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def search_for_elem_in_image(model, img_path, threshold, output_path, command):
    # Load image into a TF Tensor
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run detection
    detection = model(input_tensor)

    detection_boxes = detection['detection_boxes'][0]
    detection_classes = detection['detection_classes'][0]
    detection_scores = detection['detection_scores'][0]

    # Change the wanted element into an id
    # Potencial ids of the wanted element
    elem_id = []
    command_label = int_from_label(command["element"])
    elem_id.append(command_label)

    # Save all detected elements that fullfil the requirements to check later
    potential_elements = []

    # Iterate over all found elements
    for box, class_id, score in zip(detection_boxes, detection_classes, detection_scores):
        if ((score > threshold) == True) & (int(class_id) in elem_id):
            ymin, xmin, ymax, xmax = box
            ymin = int(ymin * height)
            xmin = int(xmin * width)
            ymax = int(ymax * height)
            xmax = int(xmax * width)

            elem = {
                "ymin" : ymin,
                "xmin" : xmin,
                "ymax" : ymax,
                "xmax" : xmax,
                "id" : class_id,
                "score" : score,
            }

            potential_elements.append(elem)
        else:
            pass
    
    if len(potential_elements) == 0:
        print("No suitable element found in the image")
    elif len(potential_elements) == 1:
        draw_bounding_box(
            potential_elements[0]["ymin"],
            potential_elements[0]["xmin"],
            potential_elements[0]["ymax"],
            potential_elements[0]["xmax"],
            potential_elements[0]["id"],
            potential_elements[0]["score"],
            image,
            ""
        )
    else:
        # Check through potential matches

        # Perform OCR
        for elem in potential_elements:
            crop_img = image[elem["ymin"]:elem["ymax"], elem["xmin"]:elem["xmax"]]
            crop_text = pytesseract.image_to_string(crop_img)

            if command["text"] in crop_text:
                draw_bounding_box(
                    elem["ymin"],
                    elem["xmin"],
                    elem["ymax"],
                    elem["xmax"],
                    elem["id"],
                    elem["score"],
                    image,
                    command["text"]
                )


            # if action == "Click":
            #     x, y = find_center(xmin, ymin, xmax, ymax)
            #     pyautogui.moveTo(x, y, 1)
            #     pyautogui.mouseDown(button = "left")
            # elif action == "Move":
            #     x, y = find_center(xmin, ymin, xmax, ymax)
            #     pyautogui.moveTo(x, y, 1) 
    cv2.imwrite(output_path, image)   

def search_for_elem_in_screenshot(model, screenshot, threshold, output_path, command):
    """
    Documentation
    """
    # Load image into a TF Tensor
    # image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    image = cv2.imread(screenshot)
    height, width, _ = image.shape
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run detection
    detection = model(input_tensor)

    detection_boxes = detection['detection_boxes'][0]
    detection_classes = detection['detection_classes'][0]
    detection_scores = detection['detection_scores'][0]

    # If the wanted element is something that the object detection could find
    if command["objectdetection?"] == True:

        # Change the wanted element into an id
        # Button could be a black or blue button (standard colors)
        if command["element"] == "Button":
            elem_id = [6, 10]
        else:
            elem_id = [int_from_label(command["element"])]

        # Save all detected elements that fullfil the requirements to check later
        potential_elements = []

        # Iterate over all found elements
        for box, class_id, score in zip(detection_boxes, detection_classes, detection_scores):
            # Look for wanted element
            if (score > threshold) and (int(class_id) in elem_id):
                ymin, xmin, ymax, xmax = box
                ymin = int(ymin * height)
                xmin = int(xmin * width)
                ymax = int(ymax * height)
                xmax = int(xmax * width)

                elem = {
                    "ymin" : ymin,
                    "xmin" : xmin,
                    "ymax" : ymax,
                    "xmax" : xmax,
                    "id" : class_id,
                    "score" : score,
                }

                potential_elements.append(elem)
            # Look for screen element for further steps
            elif (((score > threshold) == True) and (int(class_id) == 3)):
                ymin, xmin, ymax, xmax = box
                ymin = int(ymin * height)
                xmin = int(xmin * width)
                ymax = int(ymax * height)
                xmax = int(xmax * width) 

                screen_data = {
                    "ymin" : ymin,
                    "xmin" : xmin,
                    "ymax" : ymax,
                    "xmax" : xmax,
                    "id" : class_id,
                    "score" : score,
                }
            # Check if a Pop-up-window is visible
            elif (((score > threshold) == True) and (int(class_id) == 14)):
                ymin, xmin, ymax, xmax = box
                ymin = int(ymin * height)
                xmin = int(xmin * width)
                ymax = int(ymax * height)
                xmax = int(xmax * width) 

                popupwindow_data = {
                    "ymin" : ymin,
                    "xmin" : xmin,
                    "ymax" : ymax,
                    "xmax" : xmax,
                    "id" : class_id,
                    "score" : score,
                }             
            else:
                pass
        
        # Check if small elements can be detected with OCR
        if len(potential_elements) == 0:
            if (command["element"] in ["xIcon", "ArrowLeft", "ArrowRight", "InfoIcon", "SmallInfoIcon"]) or (command["text"] != ""):

                # Check if a pop-up-window is open
                try:
                    popupwindow = image[popupwindow_data["ymin"]:popupwindow_data["ymax"], popupwindow_data["xmin"]:popupwindow_data["xmax"]]
                    if command["element"] == "xIcon":
                        char = ["x", "X"]
                    elif command["element"] == "ArrowLeft":
                        char = ["<"]
                    elif command["element"] == "ArrowRight":
                        char = [">"]
                    elif command["element"] == "InfoIcon" or "SmallInfoIcon":
                        char = ["i"]
                    else:
                        char = [command["text"]]

                    data_on_screen = pytesseract.image_to_data(screen, output_type=Output.DICT)
                    n_boxes = len(data_on_screen['level'])
                    counter = 0
                    potential_elements = []

                    for i in range(n_boxes):
                        # Only check not empty strings
                        if(data_on_screen['text'][i] != ""):
                            # Wanted string was found in the given sub string
                            if char[0] in data_on_screen['text'][i]:
                                counter = counter + 1
                                (x, y, w, h) = (data_on_screen['left'][i], data_on_screen['top'][i], data_on_screen['width'][i], data_on_screen['height'][i])
                                xmin = x
                                ymin = y
                                xmax = x+w
                                ymax = y+h

                                elem = {
                                    "xmin" : xmin,
                                    "ymin" : ymin,
                                    "xmax" : xmax,
                                    "ymax" : ymax
                                }

                                potential_elements.append(elem)

                    if counter == 1:
                        cv2.rectangle(screen, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                        if command["action"] == "click":
                            x,y = find_center(ymin, xmin, ymax, xmax)
                            move_and_click(x, y)    

                    elif counter == 0:
                        print("No element found")
                        return 0

                    else:
                        pass                

                    
                except:
                    print("No PopUpWindow is visible")
            if command["text"] == "":
                print("No suitable element found in the image")
                """TODO"""
            else:
                screen = image[screen_data["ymin"]:screen_data["ymax"], screen_data["xmin"]:screen_data["xmax"]]
                # Detect all text on the screen
                text_on_screen = pytesseract.image_to_string(screen)
                # If wanted string is visible in the current screent
                if command["text"] in text_on_screen:
                    # More cranular search
                    data_on_screen = pytesseract.image_to_data(screen, output_type=Output.DICT)
                    n_boxes = len(data_on_screen['level'])

                    for i in range(n_boxes):
                        # Only check not empty strings
                        if(data_on_screen['text'][i] != ""):
                            # Wanted string was found in the given sub string
                            if command["text"] in data_on_screen['text'][i]:
                                (x, y, w, h) = (data_on_screen['left'][i], data_on_screen['top'][i], data_on_screen['width'][i], data_on_screen['height'][i])
                                xmin = x
                                ymin = y
                                xmax = x+w
                                ymax = y+h
                                cv2.rectangle(screen, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                                cv2.imwrite(output_path, screen) 

                                if command["action"] == "click":
                                    x,y = find_center(ymin, xmin, ymax, xmax)
                                    move_and_click(x, y)

        # Element is unique on screen             
        elif len(potential_elements) == 1:
            ymin = potential_elements[0]["ymin"]
            xmin = potential_elements[0]["xmin"]
            ymax = potential_elements[0]["ymax"]
            xmax = potential_elements[0]["xmax"]
            draw_bounding_box(
                ymin, xmin, ymax, xmax,
                potential_elements[0]["id"],
                potential_elements[0]["score"],
                image,
                ""
            )
            cv2.imwrite(output_path, image)

            input = {
                "ymin": ymin,
                "xmin": xmin,
                "ymax": ymax,
                "xmax": xmax,
                "action": command["action"]
            }

            if command["action"] == "click":
                x, y = find_center(ymin, xmin, ymax, xmax)
                move_and_click(x,y)
            elif command["action"] == "scroll_horizontal":
                pass
            elif command["action"] == "scroll_vertikal":
                pass


            return 1

        # Multiple potential elements are found in the screenshot
        else:
            # Check through potential matches and perform OCR
            counter = 0
            for elem in potential_elements:
                crop_img = image[elem["ymin"]:elem["ymax"], elem["xmin"]:elem["xmax"]]
                crop_text = pytesseract.image_to_string(crop_img)

                if command["text"] in crop_text:
                    draw_bounding_box(
                        elem["ymin"],
                        elem["xmin"],
                        elem["ymax"],
                        elem["xmax"],
                        elem["id"],
                        elem["score"],
                        image,
                        command["text"]
                    )
                    counter = counter + 1
            
            if counter == 1:
                if command["action"] == "click":
                    pass
                    


                # if action == "Click":
                #     x, y = find_center(xmin, ymin, xmax, ymax)
                #     pyautogui.moveTo(x, y, 1)
                #     pyautogui.mouseDown(button = "left")
                # elif action == "Move":
                #     x, y = find_center(xmin, ymin, xmax, ymax)
                #     pyautogui.moveTo(x, y, 1)
    elif command["objectdetection?"] == False:
        # Perform object detection to crop the screenshot to the simulation screen
        for box, class_id, score in zip(detection_boxes, detection_classes, detection_scores):
            # 3 is the id for screen
            if ((score > threshold) == True) & (int(class_id) == 3):
                ymin, xmin, ymax, xmax = box
                ymin = int(ymin * height)
                xmin = int(xmin * width)
                ymax = int(ymax * height)
                xmax = int(xmax * width)
        # Crop screenshot to detected screen
        screen = image[ymin:ymax, xmin:xmax]

        # List to save potential elements for later check
        potential_elements = []

        # Detect all text on the screen
        text_on_screen = pytesseract.image_to_string(screen)
        # If wanted string is visible in the current screent
        if command["text"] in text_on_screen:
            # More cranular search
            data_on_screen = pytesseract.image_to_data(screen, output_type=Output.DICT)
            n_boxes = len(data_on_screen['level'])

            for i in range(n_boxes):
                # Only check not empty strings
                if(data_on_screen['text'][i] != ""):
                    # Wanted string was found in the given sub string
                    if command["text"] in data_on_screen['text'][i]:
                        (x, y, w, h) = (data_on_screen['left'][i], data_on_screen['top'][i], data_on_screen['width'][i], data_on_screen['height'][i])
                        xmin = x
                        ymin = y
                        xmax = x+w
                        ymax = y+h

                        elem = {
                            "ymin" : ymin,
                            "xmin" : xmin,
                            "ymax" : ymax,
                            "xmax" : xmax,
                        }

                        potential_elements.append(elem)

        
                        cv2.rectangle(screen, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                        if command["action"] == "click":
                            x,y = find_center(ymin, xmin, ymax, xmax)
                            move_and_click(x, y)
            if len(potential_elements) == 0:
                print("Nothing found")
            elif len(potential_elements) == 1:
                pass
            else:
                pass
        else:
            print("Text not found")
    else:
        print("Invalid command")
    cv2.imwrite(output_path, image)   

def connect_to_simulation(process_id):
    app = Application().connect(process=process_id)
    dlg = app.top_window()
    return dlg

def click_left_button():
    pyautogui.mouseDown(button='left')
    pyautogui.mouseUp(button = "left")

def drag_mouse(xstart, ystart, xend, yend):
    pyautogui.moveTo(xstart, ystart)
    pyautogui.mouseDown(button = "left")
    pyautogui.moveTo(xend, yend, duration = 2)
    pyautogui.mouseUp(button = "left")
    time.sleep(2)

def move_and_click(x, y):
    pyautogui.moveTo(x,y, duration = 2)
    pyautogui.mouseDown(button = "left")
    pyautogui.mouseUp(button = "left")

def change_mockup(num):
    pass

def run_testscript(process_id, script, testrun_name):
    # dlg = connect_to_simulation(process_id)
    # dlg.set_focus

    change_mockup(script["mockup"])

    #model = load_model("inference_graph_efficientdet_d0_960x540/saved_model")

    for counter, command in enumerate(script["commands"]):
        screenshot = pyautogui.screenshot()
        output_path = str(testrun_name) + "/" + str(counter + 1) + ".jpg"
        print(output_path)
        #search_for_elem_in_screenshot(model, screenshot, 0.6, output_path, command)

def create_command(boolean, element, text, action):
    command = {
        "objectdetection?" : boolean,
        "element" : element,
        "text" : text,
        "action" : action
    }
    return command

testscript = {
    "mockup" : 1,
    "commands" : [
        {
            "objectdetection?" : True,
            "element" : "InfoIcon",
            "text" : "",
            "action" : "click"
        },
        {
            "objectdetection?" : True,
            "element" : "PopUpWindow",
            "text" : "",
            "action" : "spamClick"
        },
        {
            "objectdetection?" : False,
            "element" : "PopUpWindow",
            "text" : "dev",
            "action" : "click"
        },
        {
            "objectdetection?" : False,
            "element" : "PopUpWindow",
            "text" : "",
            "action" : ""
        }
    ]
}

print("Start")

#run_testscript(0, testscript, "Test")

img_path = "000007.jpg"
img = cv2.imread(img_path)
data_on_screen = pytesseract.image_to_data(img, output_type=Output.DICT)
n_boxes = len(data_on_screen['level'])
counter = 0
potential_elements = []

for i in range(n_boxes):
# Only check not empty strings
    if(data_on_screen['text'][i] != ""):
        print(data_on_screen['text'][i])

# d = pytesseract.pytesseract.image_to_string(img)
# print(d)

# model_path = "inference_graph_efficientdet_d0_960x540/saved_model"
# model = load_model(model_path)
# img_path = "Fullscreen_Screenshots/000019.jpg"
# screenshot = pyautogui.screenshot()
# # detect_on_image(model, img_path, 0.6, "Pic.jpg")
# command = {
#             "objectdetection?" : True,
#             "element" : "InfoIcon",
#             "text" : "",
#             "action" : "click"
#         }

# search_for_elem_in_screenshot(model, img_path, 0.6, "Pic_search.jpg", command)


print("Done")


# dlg = connect_to_simulation(10004)
# dlg.set_focus()

# drag_mouse(1073, 621, 600, 621)

# move_and_click(992, 587)

# move_and_click(684,604)

# time.sleep(5)

# screenshot = pyautogui.screenshot()
# print(type(screenshot))
# opencvImage = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# model_path = "Models/inference_graph_d2_960x540_V2/content/inference_graph_d2_960x540/saved_model"
# model = load_model(model_path)
# print(type(model))

# command = {
#     "element" : "InfoIcon",
#     "text" : "",
#     "action" : "click"
# }

# search_for_elem_in_screenshot(model, opencvImage, 0.6, "Res2.jpg", command)

# pyautogui.moveTo(986, 515, duration = 2)
# for i in range(10):
#     click_left_button()


# model_path = "Models/inference_graph_d2_960x540_V2/content/inference_graph_d2_960x540/saved_model"
# model_path_rcnn = "Models/inference_graph_rcnn_1024x576/content/inference_graph_rcnn_1024x576/saved_model"
# model = load_model(model_path)
# img_path = "Fullscreen_Screenshots/000019.jpg"
# detect_on_image(model, img_path, 0.6, "Pic.jpg")
# command = {
#     "element" : "Tile",
#     "text" : "Warranty",
#     "action" : "click"
# }
# search_for_elem(model, img_path, 0.6, "Pic_Find.jpg", command)