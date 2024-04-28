class Temp:
    _not_shared: str

    _not_shared_with_default: str | None

    def __init__(self, not_shared: str) -> None:
        self._not_shared = not_shared

        self._not_shared_with_default = None

if __name__ == "__main__":
    temp = Temp("test1")
    temp2 = Temp("test2")

    print(temp._not_shared)  # prints "test1"
    print(temp2._not_shared) # prints "test2"

    print(temp._not_shared_with_default)  # prints None
    print(temp2._not_shared_with_default) # prints None

    temp._not_shared = "Some third thing"
    temp._not_shared_with_default = "Not shared"

    print(temp._not_shared)  # prints "Some third thing"
    print(temp2._not_shared) # prints "test2"

    print(temp._not_shared_with_default)  # prints "Not shared"
    print(temp2._not_shared_with_default) # prints None
