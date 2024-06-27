import json

class ObjPred:
    def __init__(self, label, confidence, color, rectangle):
        self.label = label
        self.confidence = confidence
        self.color = color
        self.rectangle = rectangle

    def to_json(self):
        return {
            'label': self.label,
            'confidence': self.confidence,
            'color': self.color,
            'rectangle': self.rectangle
        }

    def write_to_file(self, new_data):
        file_path = 'src/app_core/models/output.txt'

        try:
            with open(file_path, 'r') as file:
                existing_content = file.read()
            data_list = json.loads(existing_content)
        except (FileNotFoundError, json.JSONDecodeError):
            data_list = []

        data_list.append(new_data)

        updated_content = json.dumps(data_list, indent=4)

        with open(file_path, 'w') as file:
            file.write(updated_content)

