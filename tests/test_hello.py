def hello_wolrd():
    print("hello world")
    return True


def test_HelloWorld():
    assert hello_wolrd() == True

if __name__ == "__main__":
    hello_wolrd()  # This prints "hello world" when running `python test_script.py`
