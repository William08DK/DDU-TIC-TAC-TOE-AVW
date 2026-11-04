from ultralytics import YOLO
import os

DIR = os.path.dirname(os.path.abspath(__file__))
# Husk at opdatere "train" hver gang du har trænet en ny model
MODEL = os.path.join(DIR, "runs", "detect", "train", "weights", "best.pt")
IMAGE = os.path.join(DIR, "input-test.jpg")
ROWS, COLS = 3, 5

LETTERS = {
    "circle": "C",  # cirkel
    "maltese": "M",  # maltesisk kors
    "square": "F",  # firkant
    "star": "S",  # stjerne
    "temple": "T",  # tempel
    "nothing": "-",  # tomt felt
}

model = YOLO(MODEL)
result = model(IMAGE)[0]

# Brættet som dictionary: nøglen er (række, kolonne), værdien er (navn, sikkerhed)
# Felter uden symbol oprettes ikke i dict.

# Bemærk - algoritmen har en svaghed - hvis brættet ikke udfylder hele billedet, hvorfor?
board = {}

for box in result.boxes:
    name = result.names[int(box.cls)]  # klassens navn, fx "star"
    conf = float(box.conf)  # confidence

    # Midten normaliseret (0-1) – bruges til at finde feltet på brættet
    # Bemærk attribut "xywhn" - se dokumentation!
    x, y, w, h = box.xywhn[0].tolist()
    row = min(
        int(y * ROWS), ROWS - 1
    )  # min(...) så en midte på kanten (1.0) ikke ryger udenfor
    col = min(int(x * COLS), COLS - 1)

    # Konflikt: vælg det med højest confidence
    if (row, col) not in board or conf > board[(row, col)][1]:
        board[(row, col)] = (name, conf)

# Print brættet række for række
for row in range(ROWS):
    line = ""
    for col in range(COLS):
        if (row, col) in board:
            name = board[(row, col)][0]
            line += LETTERS.get(name, "?") + " "  # "?" hvis klassen ikke står i LETTERS
        else:
            line += "- "
    print(line)
