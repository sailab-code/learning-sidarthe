import string

import config
import os.path as path
import json

from record import Record


class DataManager:

    def __init__(self):

        self.repo_path = path.join(path.curdir, config.REPO_NAME)
        if not path.exists(self.repo_path) or not path.isdir(self.repo_path):
            raise FileNotFoundError("Cannot find data repository")

    def __get_data_from_json_file(self, filename: string):
        file_path = path.join(self.repo_path, config.JSON_DATA_PATH, f"{config.DATA_FILE_PREFIX}{filename}.json")
        if not path.exists(file_path):
            raise FileNotFoundError("Cannot find data file")

        file = open(file_path, "r")
        data = json.load(file)
        file.close()
        return data

    def get_latest_national(self):
        data = self.__get_data_from_json_file("ita-andamento-nazionale-latest")
        return Record(data[0])

    def get_all_national(self):
        data = self.__get_data_from_json_file("ita-andamento-nazionale")
        records = []
        for record in data:
            records.append(Record(record))
        return records

    def get_latest_provinces(self):
        data = self.__get_data_from_json_file("ita-province-latest")
        return Record(data[0])

    def get_all_provinces(self):
        data = self.__get_data_from_json_file("ita-province")
        records = []
        for record in data:
            records.append(Record(record))
        return records

    def get_latest_regions(self):
        data = self.__get_data_from_json_file("ita-regioni-latest")
        return Record(data[0])

    def get_all_regions(self):
        data = self.__get_data_from_json_file("ita-regioni")
        records = []
        for record in data:
            records.append(Record(record))
        return records
