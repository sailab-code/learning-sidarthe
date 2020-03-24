import json


class Record:
    def __init__(self, json_data):
        self.date = json_data["data"]
        self.nation = json_data["stato"]
        self.total_hospitalized = json_data["totale_ospedalizzati"]
        self.hospitalized_with_symptoms = json_data["ricoverati_con_sintomi"]
        self.intensive_care = json_data["terapia_intensiva"]
        self.home_isolation = json_data["isolamento_domiciliare"]
        self.total_positives = json_data["totale_attualmente_positivi"]
        self.new_positives = json_data["nuovi_attualmente_positivi"]
        self.recovered = json_data["dimessi_guariti"]
        self.deceased = json_data["deceduti"]
        self.total_cases = json_data["totale_casi"]
        self.swabs = json_data["tamponi"]

    def __repr__(self):
        return json.dumps(self.__dict__)