"""Find vores fysiske 3 x 3-bræt ud fra kanten og de ni firkantede felter."""

import cv2
import numpy as np

# Denne fil finder PLADEN, ikke kryds og boller. YOLO-modellen genkender brikkerne
# et andet sted i programmet. Her bruger vi kanter, farver og formen på gitteret.
#
# Overblik over forløbet:
# 1. Find mulige brætter ud fra trærammen og de blå felter.
# 2. Kontrollér, at hvert forslag faktisk ligner et 3 x 3-bræt.
# 3. Vælg det bedste forslag og vent på en stabil placering.
# 4. Behold zoomet ved støj; flyt det først efter en bekræftet flytning.
#
# Et frame er ét kamerabillede, gemt som en tabel af pixels. I billedkoordinater
# er (0, 0) øverst til venstre. x vokser mod højre, og y vokser nedad.
# None betyder "ingen sikker værdi endnu" og er ikke et sæt hjørner.


def ordered_corners(points):
    """Rækkefølge: øverst venstre, øverst højre, nederst højre, nederst venstre."""
    # OpenCV kan give hjørner i forskellig rækkefølge. Vi skal altid starte det
    # samme sted, ellers kan det oprettede bræt pludselig vende på hovedet.
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    # Middelværdien af de fire hjørner er firkantens centrum.
    center = points.mean(axis=0)
    # Beregn retningen fra centrum til hvert hjørne og sortér dem rundt om centrum.
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    # Hjørnet med lavest x+y vælges som øverst venstre. roll flytter rækkefølgen,
    # men ændrer ikke selve koordinaterne. Det passer til vores omtrent oprette bræt.
    return np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)


def grid_evidence(frame, corners):
    """En firkant er ikke nok: kontrollér, at den indeholder et 3 x 3-gitter."""
    # Ret forslaget op til samme størrelse hver gang. Så kan vi lede efter
    # felter på kendte steder, selvom brættet er skråt eller langt fra kameraet.
    size = 300
    target = np.float32([(0, 0), (size, 0), (size, size), (0, size)])
    matrix = cv2.getPerspectiveTransform(corners, target)
    board = cv2.warpPerspective(frame, matrix, (size, size))
    # Gråtoner gør farver unødvendige i kantkontrollen. En lille sløring dæmper
    # pixelstøj, hvorefter Canny markerer tydelige ændringer i lysstyrke som kanter.
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 15, 45)
    # CLOSE lukker små huller mellem kantstykker. 3 x 3 er størrelsen på det
    # lille område, OpenCV bruger til at forbinde kanterne.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # En kontur er en sammenhængende omridskurve i kantbilledet.
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Et set tæller hvert felt én gang, selvom både inder- og yderkanten findes.
    cells = set()
    for contour in contours:
        area = cv2.contourArea(contour)
        # Et felt er omtrent 100 x 100 pixels i vores oprettede 300 x 300-billede.
        # Spring små detaljer og meget store former over.
        if not 2200 < area < 12500:
            continue
        # Forenkl en lang kantkurve til nogle få hjørner. Fire hjørner og en
        # udadbuet figur (konveks) er et muligt firkantet felt.
        polygon = cv2.approxPolyDP(contour, .035 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        x, y, w, h = cv2.boundingRect(polygon)
        # Bredden og højden skal ligne et kvadrat; en lang, tynd stribe er ikke et felt.
        if not .65 < w / h < 1.55:
            continue
        cx, cy = x + w / 2, y + h / 2
        # Del billedet i tre dele hver vej. // er heltalsdivision: 150 // 100 = 1.
        col, row = int(cx // 100), int(cy // 100)
        if 0 <= row < 3 and 0 <= col < 3:
            # Centrum skal passe til et af de ni felter, ikke ligge på en gitterlinje.
            if abs(cx - (col * 100 + 50)) < 26 and abs(cy - (row * 100 + 50)) < 26:
                cells.add((row, col))
    # Brikker kan røre feltkanten og bryde en kontur. De lange gitterlinjer
    # giver ekstra støtte, men en vilkårlig firkant uden felter godkendes aldrig.
    # OPEN med en lang, smal form beholder især lodrette eller vandrette kantstykker.
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((30, 1), np.uint8))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((1, 30), np.uint8))
    # Gitterets indre skillelinjer forventes omkring pixel 100 og 200.
    # Se i en smal stribe omkring hvert sted, og kræv støtte langs mere end 45 %
    # af billedets højde/bredde. Vi tillader altså mellemrum i en gitterlinje.
    lines = sum(np.mean(np.any(vertical[:, p - 12:p + 13], axis=1)) > .45 for p in (100, 200))
    lines += sum(np.mean(np.any(horizontal[p - 12:p + 13, :], axis=0)) > .45 for p in (100, 200))
    # Dette er point, ikke en procent: ét point pr. felt og fire ekstra point,
    # hvis mindst tre felter og alle fire indre skillelinjer støtter forslaget.
    return len(cells) + (4 if len(cells) >= 3 and lines == 4 else 0)


def blue_grid_candidates(frame):
    """Brug de ni blå felter/brikker, når trærammens kant er brudt i billedet."""
    # På den aktuelle plade er de tomme felter blå. Deres centre er lettere at
    # følge end trærammen, som kan flyde sammen med baggrund og løse træstykker.
    # HSV beskriver farvetone, farvemætning og lysstyrke. Det gør det lettere at
    # udvælge et interval af blå farver end ved at sammenligne rød/grøn/blå direkte.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (65, 25, 45), (120, 255, 255))
    # Fjern små blå støjpletter, før vi leder efter sammenhængende blå områder.
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = frame.shape[0] * frame.shape[1]
    centers = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        # Et område må hverken være en lille refleks eller det meste af baggrunden.
        if not area * .0015 < cv2.contourArea(contour) < area * .13:
            continue
        center, (width, height), _ = cv2.minAreaRect(contour)
        # Find en firkant omkring området, også hvis den står skråt.
        # Vi bruger midten og afviser meget små eller meget aflange områder.
        if min(width, height) >= 8 and .5 < width / height < 2:
            centers.append(center)
    # Begræns arbejdet, hvis der er mange blå genstande i billedet.
    centers = np.float32(centers[:24])
    if len(centers) < 9:
        # Denne metode kræver ni centre. Rammesøgningen kan stadig give et forslag.
        return []
    # Her er de ni FELTCENTRE på et tænkt bræt med bredde og højde 1.
    # Første centrum ligger ved (1/6, 1/6), næste ved (1/2, 1/6) osv.
    grid = np.float32([((col + .5) / 3, (row + .5) / 3)
                       for row in range(3) for col in range(3)])
    proposals, seen = [], set()
    # Prøv lokale grupper, så en blå genstand uden for brættet ikke ødelægger søgningen.
    for center in centers:
        # norm beregner afstand. argsort sorterer fra nærmest til længst væk.
        # Vælg de ni nærmeste centre og husk gruppen, så den ikke prøves flere gange.
        indices = tuple(sorted(np.argsort(np.linalg.norm(centers - center, axis=1))[:9]))
        if indices in seen:
            continue
        seen.add(indices)
        points = centers[list(indices)]
        # Brug gruppens ydre firkant til at finde retningerne "hen over" og "ned ad" brættet.
        bounds = ordered_corners(cv2.boxPoints(cv2.minAreaRect(points)))
        across, down = bounds[1] - bounds[0], bounds[3] - bounds[0]
        if min(np.linalg.norm(across), np.linalg.norm(down)) < 20:
            continue
        across /= np.linalg.norm(across)
        down /= np.linalg.norm(down)
        # @ måler her, hvor langt hvert punkt ligger i en bestemt retning.
        # Sortér først i tre rækker og derefter fra venstre mod højre i hver række.
        rows = np.split(points[np.argsort(points @ down)], 3)
        ordered = np.concatenate([row[np.argsort(row @ across)] for row in rows])
        # Beregn en opskrift, som flytter de tænkte ni centre til de ni fundne centre.
        # Den samme opskrift kan bagefter fortælle os, hvor de fire yderhjørner ligger.
        matrix, _ = cv2.findHomography(grid, ordered, 0)
        if matrix is None:
            continue
        corners = cv2.perspectiveTransform(np.float32([[(0, 0), (1, 0), (1, 1), (0, 1)]]), matrix)[0]
        projected = cv2.perspectiveTransform(grid.reshape(1, -1, 2), matrix)[0]
        # projected er de steder, opskriften forventer. De skal passe til de faktiske
        # centre; ellers har vi måske samlet ni tilfældige blå genstande.
        board_area = cv2.contourArea(corners)
        if not area * .07 < board_area < area * .90 or not cv2.isContourConvex(corners):
            continue
        # De ni centre skal passe til et regelmæssigt 3 x 3-bræt med meget lille fejl.
        if np.max(np.linalg.norm(projected - ordered, axis=1)) > np.sqrt(board_area) * .04:
            continue
        if (np.any(corners < 2) or np.any(corners[:, 0] >= frame.shape[1] - 2)
                or np.any(corners[:, 1] >= frame.shape[0] - 2)):
            # Godkend ikke et forslag, hvis selve brættet ligger uden for billedet.
            continue
        proposals.append(corners)
    return proposals


def find_board(frame):
    """Returnér fire hjørner eller None. Kræver et bekræftet 3 x 3-gitter."""
    # Søg på en mindre kopi for at spare regnekraft. Vi gemmer skaleringen, så
    # de fundne hjørner kan omregnes til originalbilledets pixels til sidst.
    height, width = frame.shape[:2]
    scale = min(1., 720 / max(height, width))
    small = cv2.resize(frame, (round(width * scale), round(height * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Webcammet giver blødere kanter end træningsfotos. En høj tærskel bryder
    # rammen og gør især det tomme bræt usynligt, selv når det ses tydeligt.
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 15, 45)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Vores lyse træramme skiller sig også farvemæssigt fra bordet. Det hjælper,
    # når skygger eller en brik bryder den yderste kant i kantbilledet.
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    wood = cv2.inRange(hsv, (5, 18, 50), (45, 210, 255))
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    wood_contours, _ = cv2.findContours(wood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Saml forslag fra almindelige kanter og de træfarvede områder.
    contours = list(contours) + list(wood_contours)
    image_area = small.shape[0] * small.shape[1]
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        # Rammen skal fylde en rimelig del af billedet: mellem 7 % og 90 %.
        if not image_area * .07 < area < image_area * .90:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, .025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        corners = ordered_corners(polygon)
        if (np.any(corners < 2) or np.any(corners[:, 0] >= small.shape[1] - 2)
                or np.any(corners[:, 1] >= small.shape[0] - 2)):
            continue  # Hele rammen skal være synlig, ikke skåret af billedkanten.
        sides = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)
        # Afvis meget små eller stærkt aflange firkanter, som næppe er vores bræt.
        if sides.min() < 55 or sides.max() / sides.min() > 2.2:
            continue
        if any(np.max(np.linalg.norm(corners - old, axis=1)) < 8 for old in candidates):
            # Den samme ramme kan findes flere gange. Gem næsten ens forslag kun én gang.
            continue
        candidates.append(corners)
        if len(candidates) >= 10:
            # En grænse holder søgningen hurtig, selv med en rodet baggrund.
            break
    # Tilføj også brætforslagene fra de blå felter/brikker, før alle forslag bedømmes.
    candidates.extend(blue_grid_candidates(small))
    scored = [(grid_evidence(small, corners), cv2.contourArea(corners), corners)
              for corners in candidates]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # Flest gitterpoint vinder. Ved samme pointtal foretrækkes det største område.
    # Syv er en pointgrænse; det betyder ikke nødvendigvis syv synlige felter.
    if not scored or scored[0][0] < 7:
        return None
    return scored[0][2] / scale  # Tilbage til koordinater i det oprindelige kamerabillede.


class BoardTracker:
    """Lås visningen fast; hold aktuelle målinger adskilt fra den gemte zoom."""

    def __init__(self):
        self.reset()

    def reset(self):
        # corners: den GEMTE placering, som holder zoomet stille.
        # previous: sidste søgeforslag, som bruges til at bekræfte en flytning.
        # stable: hvor mange ensartede søgninger vi har set i træk.
        # ready: om den AKTUELLE placering er sikker nok til aflæsning.
        # measurement: hjørnerne i den seneste faktiske måling, ikke den gemte zoom.
        self.corners = None
        self.previous = None
        self.stable = 0
        self.ready = False
        self.measurement = None

    def mark_missing(self):
        # Et manglende fund gør aflæsning usikker, men må ikke nulstille zoomet.
        # Kun reset() (kameraskift/Find igen) sletter den allerede låste visning.
        self.previous = None
        self.stable = 0
        self.ready = False
        self.measurement = None

    def update(self, frame):
        # Kaldet sker løbende fra main.py, også efter at brættet allerede er fundet.
        found = find_board(frame)
        self.ready = False
        if found is None:
            self.mark_missing()
            return None
        if self.corners is not None:
            # Bevar felternes orientering ved mindre, sammenhængende drejninger.
            variants = [np.roll(found, i, axis=0) for i in range(4)]
            # Sammenlign hjørnernes placering i forhold til centrum. Dermed bliver
            # en forskydning ikke forvekslet med, at brættet har fået en ny top/bund.
            reference = self.corners - self.corners.mean(axis=0)
            found = min(variants, key=lambda p: np.linalg.norm((p - p.mean(axis=0)) - reference))
        self.measurement = found.copy()
        # Få pixels forskel skyldes ofte støj eller skift mellem to søgemetoder.
        # Behold da den låste zoom helt stille; aflæsning bruger stadig den nye måling.
        tolerance = max(5., min(frame.shape[:2]) * .025)
        # Tolerancen er mindst 5 pixels og vokser lidt med billedstørrelsen.
        # max(norm(...)) ser på det hjørne, der har flyttet sig mest.
        if self.corners is not None and np.max(np.linalg.norm(found - self.corners, axis=1)) <= tolerance:
            self.previous = found
            self.stable = 3
            self.ready = True
            return self.corners.copy()
        # En reel flytning skal bekræftes flere gange. Indtil da beholdes den gamle
        # visning, men ready er falsk, så gamle koordinater ikke kan godkende et træk.
        if self.previous is not None and np.max(np.linalg.norm(found - self.previous, axis=1)) < tolerance:
            # Det nye forslag ligner det forrige: endnu en bekræftelse.
            self.stable += 1
        else:
            # Forslagene er uenige, så begynd tællingen for den nye placering forfra.
            self.stable = 1
        self.previous = found
        if self.stable < 3:
            return None
        # Først efter tre ensartede søgninger flyttes den gemte zoom til det nye sted.
        self.corners = found
        self.ready = True
        return self.corners.copy()
