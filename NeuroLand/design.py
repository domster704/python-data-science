import sys
from PyQt5 import uic, QtWidgets
from PyQt5 import QtCore
from PythonDataScience.NeuroLand.programs_parser import domofond_parser
from PyQt5.QtWidgets import QMessageBox


#  https://www.domofond.ru/uchastokzemli-na-prodazhu-skoropuskovskiy-1648805439

class UI(QtWidgets.QMainWindow):
	def __init__(self):
		super(UI, self).__init__()
		uic.loadUi('ui/ver4.ui', self)
		self.link = str()
		self.data = str()
		self.city = {
			"Камчатский край": "1",
			"Марий Эл": "2",
			"Чечня": "3",
			"Оренбургская область": "4",
			"Ямало-Ненецкий АО": "5",
			"Забайкальский край": "6",
			"Ярославская область": "7",
			"Владимирская область": "8",
			"Бурятия": "9",
			"Калмыкия": "10",
			"Белгородская область": "11",
			"Вологодская область": "12",
			"Волгоградская область": "13",
			"Калужская область": "14",
			"Ингушетия": "15",
			"Кабардино-Балкария": "16",
			"Иркутская область": "17",
			"Ивановская область": "18",
			"Астраханская область": "19",
			"Карачаево-Черкесия": "20",
			"Новгородская область": "21",
			"Курганская область": "22",
			"Костромская область": "23",
			"Краснодарский край": "24",
			"Магаданская область": "25",
			"Нижегородская область": "26",
			"Кировская область": "27",
			"Липецкая область": "28",
			"Мурманская область": "29",
			"Курская область": "30",
			"Мордовия": "31",
			"Хакасия": "32",
			"Карелия": "33",
			"Якутия": "34",
			"Татарстан": "35",
			"Адыгея": "36",
			"Омская область": "37",
			"Пензенская область": "38",
			"Псковская область": "39",
			"Северная Осетия": "40",
			"Башкортостан": "41",
			"Пермский край": "42",
			"Ростовская область": "43",
			"Дагестан": "44",
			"Приморский край": "45",
			"Орловская область": "46",
			"Томская область": "47",
			"Тверская область": "48",
			"Удмуртия": "49",
			"Ставропольский край": "50",
			"Ульяновская область": "51",
			"Хабаровский край": "52",
			"Смоленская область": "53",
			"Ханты-Мансийский АО": "54",
			"Челябинская область": "55",
			"Самарская область": "56",
			"Тульская область": "57",
			"Тамбовская область": "58",
			"Тюменская область": "59",
			"Свердловская область": "60",
			"Сахалинская область": "61",
			"Рязанская область": "62",
			"Республика Алтай": "63",
			"Чувашия": "64",
			"Чукотский АО": "65",
			"Брянская область": "66",
			"Еврейская АО": "67",
			"Алтайский край": "68",
			"Калининградская область": "69",
			"Архангельская область": "70",
			"Кемеровская область": "71",
			"Амурская область": "72",
			"Воронежская область": "73",
			"Красноярский край": "74",
			"Ненецкий АО": "75",
			"Тыва": "76",
			"Коми": "77",
			"Новосибирская область": "78",
			"Саратовская область": "79",
			"Ленинградская область": "80",
			"Московская область": "81",
			"Крым": "82",
		}

	def get_link(self):
		try:
			self.website_cost.setText('')
			self.data = domofond_parser.get_data_by_link(self.lineEdit.text().replace(' ', ''))
			self.data[-1] = self.city[self.data[-1]]
			print(self.data)

			# self.data = [7.0, 10.0, 600000.0, 3.4, 3.2, 3.0, 3.7, 3.5, 3.1, 4.2, 3.2, 3.5, 2.3, 49]

			from PythonDataScience.NeuroLand.tensorflow_neuro import set_data_from_design
			result = set_data_from_design(self.data)

			self.result.setText(str(int(result)) + "₽")
			self.cost1.setText("Cost")
			self.cost2.setText("Cost on website")
			self.website_cost.setText(str(int(self.data[2])) + " ₽")
		except Exception as e:
			QMessageBox.warning(self, 'Error', "Try again", QMessageBox.Ok)
			print(e)

	def keyPressEvent(self, qKeyEvent):
		if qKeyEvent.key() == QtCore.Qt.Key_Return:
			self.get_link()

	# Формат данных в data:
	# '[10сот', '4км', '570000', Эколгоия: 3.1, Чистота: 2.9, ЖКХ: 2.7, Соседи: 3.7,
	# Условия для детей: 3.5, Спорт и отдых: 3.1, Магазины: 4.2, Транспорт: 2.9, Безопасность: 3.2]
	def initUI(self):
		self.setFixedSize(800, 600)
		self.setWindowTitle("NeuroLand")
		self.lineEdit.setText(' ')
		self.lineEdit.setText('https://www.domofond.ru/uchastokzemli-na-prodazhu-skoropuskovskiy-1648805439')
		self.find_button.clicked.connect(self.get_link)

		self.show()


if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	root = UI()
	root.initUI()
	app.exec_()
