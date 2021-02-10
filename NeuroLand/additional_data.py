import re


def get_average(filename='results/obhiy'):
    average = [0 for i in range(10)]
    var = 0
    amount = 1
    with open(f'{filename}.csv', 'r', encoding='utf-8') as f:
        for i in f:
            if var == 0:
                var = 1
                continue
            amount += 1
            dictionary = list(map(float, [x for x in i.split(',')[3:-1]]))
            for j in range(10):
                average[j] += dictionary[j]
        for i in range(len(average)):
            average[i] = round(average[i] / amount, 1)
        print(average)
    with open('results/average_values.txt', 'w', encoding='utf-8') as f:
        f.write(','.join(list(map(str, average))))


def get_average_from_file():
    with open('results/average_values.txt', 'r', encoding='utf-8') as f:
        return f.read()


if __name__ == "__main__":
    get_average()