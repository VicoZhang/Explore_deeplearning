def a(m, *x, **y):
    print(m)
    print(x)
    print(y)


a(45)
print('--------------')
a(1, 2, 3, b=1, c=4)
