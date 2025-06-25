from enum import Enum

class Photo:
    class Class(Enum):
        Acceptable = 'Acceptable'
        Unacceptable = 'Unacceptable'
        Uncategorized = 'Uncategorized'
        
    def __init__(self, path, actual):
        self.path = path
        self.actual = actual
        self.id = int(path.split('/')[-1].split('.')[0])

    def classify(self, result):
        self.result = result
        if "Acceptable" in result:
            self.predicted = self.Class.Acceptable
        elif "Unacceptable" in result:
            self.predicted = self.Class.Unacceptable
        else:
            self.predicted = self.Class.Uncategorized
        self.case = Photo.get_confusion(self.actual, self.predicted)
        # print(self.get_name())

    def get_name(self):
        return f'{self.case} {self.path} {self.result}'

    @staticmethod
    def get_confusion(actual, predicted):
        if (predicted==Photo.Class.Uncategorized):
            return 'Uncategorized'
        if (actual==Photo.Class.Acceptable):
            if (predicted==Photo.Class.Acceptable):
                return 'TP'
            else:
                return 'FN'
        else:
            if (predicted==Photo.Class.Acceptable):
                return 'FP'
            else:
                return 'TN'