def main():
    while True:
        user_input = input("문장을 입력하세요 ('!quit'을 입력하면 종료): ")

        if user_input == '!quit':
            print("프로그램을 종료합니다.")
            break
        
        print("입력한 문장:", user_input)

if __name__ == "__main__":
    main()
