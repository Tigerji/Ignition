import random

sample_week = []
weeks = 4

for _ in range(100):
    reached = 0
    gear_list = [593 for _ in range(16)]
    for i in range(weeks):
        loot1 = random.randint(0, 15)
        if gear_list[loot1] < 616:
            gear_list[loot1] = 616
        else:
            loot2 = random.randint(0, 15)
            if gear_list[loot2] < 616:
                gear_list[loot2] = 616
        for _ in range(4):
            loot = random.randint(0, 15)
            if gear_list[loot] < 606:
                gear_list[loot] = 606
            if random.random() < 0.05:
                loot = random.randint(0, 15)
                if gear_list[loot] < 610:
                    gear_list[loot] = 610
    gear_list[gear_list.index(min(gear_list))] = 606
    gear_list[gear_list.index(min(gear_list))] = 606
    sample_week.append(sum(gear_list) / len(gear_list))

sample_week = sorted(sample_week)
print(sum(sample_week) / len(sample_week), min(sample_week), max(sample_week))

# 1st time loot + books: 44
# 1st time gather: 24
# mettle: 45
# total: 113
# weekly: 11 * 4


# weekly chests opened
# 84736,84737,84738,84739