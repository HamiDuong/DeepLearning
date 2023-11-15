import pandas as pd

df = pd.read_csv("Outputs/test.csv")

# python .\detect_from_image.py -m .\inference_graphs\inference_graph_efficientdet_d3_1980x1080_V2\saved_model -l .\label_map.pbtxt -i .\test_images_v2