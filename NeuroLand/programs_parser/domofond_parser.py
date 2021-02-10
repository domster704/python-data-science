from bs4 import BeautifulSoup as bs
import requests
import re
from PythonDataScience.NeuroLand import additional_data

HEADERS = {
	'User-Agent': ''  # UserAgent(verify_ssl=False).chrome
}
CITY = {
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


def parser(start_page=1, end_page=1, page_counter=True, _headers=None):
	"""Записывает данные из domofond.ru по нескольким параметрам

	start_page(int) -- номер страницы, с которой функция начнёт парсинг
	end_page(int) -- номер страницы, на которой функция закончит парсинг
	page_counter(bool) -- показывать номер страницы, которую обрабатывает функция
	_headers -- user agent

	"""
	if _headers is None:
		_headers = HEADERS

	if start_page > end_page:
		print('Стартовая страница не может быть меньше конечной')
		exit()

	url = 'https://www.domofond.ru/prodazha-uchastkizemli-leningradskaya_oblast-r'
	counter = 1
	for region in range(1, 83):
		try:
			soup = bs(
				requests.get(f'{url}{region}?Page=1',
							 headers=_headers).content, 'html.parser')  # страница
			li = soup.findAll('li', 'pagination__page___2dfw0')
			nav = list()
			for i in range(len(li)):
				nav.append(int(li[i]['data-marker'].split('-')[1]))
			end_page = max(nav)
			print('Количество страниц: ', end_page, "Регион: ", region)
		except Exception as e:
			print(e)

		for i in range(start_page, end_page + 1):
			if page_counter:
				print('СТРАНИЦА НОМЕР: ', i)
			try:
				soup = bs(
					requests.get(f'{url}{region}?Page={i}',
								 headers=_headers).content, 'html.parser')  # страница
			except Exception as e:
				print(e)
				continue
			articles_links = soup.findAll('a', 'long-item-card__item___ubItG search-results__itemCardNotFirst___3fei6')[
							 1:]
			for link in articles_links:
				try:
					response_nested = requests.get(f'https://www.domofond.ru{link["href"]}',
												   headers=_headers)
				except Exception:
					continue
				soup_nested = bs(response_nested.content, 'html.parser')
				detail_information = soup_nested.findAll('div', 'detail-information__row___29Fu6')

				# ОЦЕНКА РАЙОНА
				ratings = {}
				try:
					for rating in soup_nested.findAll('div', 'area-rating__row___3y4HH'):
						ratings[rating.find('div', 'area-rating__label___2Y1bh').get_text()] \
							= rating.find('div', 'area-rating__score___3ERQc').get_text()
				except AttributeError:
					print('Оценка отсутствует')
					continue

				if not ratings:
					continue

				area, price = [detail.get_text().split(':')[1] for detail in detail_information[2:4]]
				proximity = detail_information[1].get_text().split(':')[1].split(',')[0]
				if re.sub(r'[₽ ]', '', price) == "Неуказано":
					continue
				else:
					price = float(re.sub(r'[₽ ]', '', price))
				area = float(re.sub(r'сот..', '', area).replace(' ', ''))
				proximity = re.sub(r'[км ]', '', proximity)

				if '.' in proximity:
					proximity = proximity.split('.')[0]
				if proximity == "Вчертегорода":
					proximity = '2'
				proximity = float(proximity)
				with open('../results/allRegions.csv', 'a', encoding='utf-8') as f:
					f.write(
						f'{area},{proximity},{price},{",".join([y.replace(",", ".") for x, y in ratings.items()])},{region}\n'.replace(
							' ', '')
					)
					print(f'записан номер {counter}, регион: {region}')
				counter += 1
				ratings.clear()


def get_data_by_link(url: str):
	"""Получает и парсит полученную ссылку

	:param url: ссылка (str)
	:return: данные об участке земли (str)

	"""

	response = requests.get(url).content
	soup_nested = bs(response, 'html.parser')
	detail_information = soup_nested.findAll('div', 'detail-information__row___29Fu6')
	city = soup_nested.find('p', 'location__text___bhjoZ').get_text().split(',')[-1].strip()

	# Характеристики земли
	area, price = [detail.get_text().split(':')[1] for detail in detail_information[2:4]]
	proximity = detail_information[1].get_text().split(':')[1].split(',')[0]
	price = re.sub(r'[₽ ]', '', price)
	area = re.sub(r'сот..', '', area).replace(' ', '')
	proximity = re.sub(r'[км ]', '', proximity)

	if '.' in proximity:
		proximity = proximity.split('.')[0]
	if proximity == "Вчертегорода":
		proximity = '2'

	# print(area, proximity, price, city)
	# Оценка района
	ratings = {}
	void_ratings = additional_data.get_average_from_file()
	dummy_response = f'{area},{proximity},{price},{void_ratings},{city}'.split(',')
	dummy_response[-1] = CITY[dummy_response[-1]]
	dummy_response = list(map(float, dummy_response))

	try:
		for rating in soup_nested.findAll('div', 'area-rating__row___3y4HH'):
			ratings[rating.find('div', 'area-rating__label___2Y1bh').get_text()] \
				= rating.find('div', 'area-rating__score___3ERQc').get_text()
	except AttributeError:
		return dummy_response

	if not ratings:
		return dummy_response
	data_ = f'{area},{proximity},{price},{",".join([y.replace(",", ".") for x, y in ratings.items()])},{CITY[city]}'.split(',')
	data_[-1] = CITY[data_[-1]]
	data_ = list(map(float, data_))
	return data_


if __name__ == "__main__":
	parser()
