import object_detection

detector = object_detection.FRCNNDetector(device="cuda", num_positions=8, roi_pool_size=64)
detector.display_bounding_boxes(image_path="../misogyny_data/img/5331.jpg", destination_path="../ausarbeitung/figures/detection/detection.pdf")