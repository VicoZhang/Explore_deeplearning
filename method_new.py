class Person:
    _instance = None

    def __init__(self, name):
        self.name = name

    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = object.__new__(cls)
    #     return cls._instance


# id_temp = []
for i in range(10):
    p = Person('{}'.format(i))
    print(p)
    print(id(p))

# print(id_temp)
# id_set = set(id_temp)
# print(id_set)
