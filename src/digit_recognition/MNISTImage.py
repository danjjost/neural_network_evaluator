class MNISTImage():
    def __init__(self, image_array: list[float], label: int):
        self.image_array = image_array
        self.label = label 