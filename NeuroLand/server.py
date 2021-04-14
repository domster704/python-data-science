from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from programs_parser.domofond_parser import get_data_by_link
import uvicorn
from new_tens import getDataFromReadyNeural


class Url(BaseModel):
	data: str


# class Data(BaseModel):
# 	area: str
# 	proximity: str
# 	ecology: str
# 	purity: str
# 	utilities: str
# 	neighbours: str
# 	children: str
# 	sport: str
# 	shop: str
# 	transport: str
# 	safety: str
# 	lifecost: str
# 	region: str

class Data(BaseModel):
	area: str
	proximity: str
	ecology: str
	purity: str
	utilities: str
	neighbors: str
	children: str
	sport: str
	shop: str
	transport: str
	safety: str
	lifecost: str
	region: str


city = {
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
app = FastAPI(title="NeuroLand")


@app.get("/", response_class=PlainTextResponse)
async def root():
	return "hello, it's test response"


# TODO: сделать функцию get_cost_by_data
@app.post("/url")
async def get_cost_by_url(item: Url):
	url = item.data
	data = get_data_by_link(url)
	per = getDataFromReadyNeural(data)
	print(per)
	return round(per, 2)


@app.post("/data")
def get_cost_by_data(info: Data):
	data = list(dict(info).values())
	data[-1] = city[data[-1]]
	data = list(map(float, data))

	print(data)
	per = getDataFromReadyNeural(data)
	print(per)
	return round(per)


@app.post("/test")
def test(info: Url):
	return info.data


if __name__ == "__main__":
	try:
		test_var = getDataFromReadyNeural(
			get_data_by_link("https://www.domofond.ru/uchastokzemli-na-prodazhu-fryazino-1766602262"))
	except:
		test_var = getDataFromReadyNeural(
			get_data_by_link("https://www.domofond.ru/uchastokzemli-na-prodazhu-fryazino-1766609629"))
	finally:
		pass
	uvicorn.run(app, host="0.0.0.0", port=8000)
