# finalsite

### 설치

`git clone https://github.com/8bitHermitcrab/finalsite.git`

### 사용법

`streamlit run main.py`

## 👤구성원

### 조장
김재헌 : [GitHub](https://github.com/gemjh)

### 조원
고인정 : [GitHub](https://github.com/8bitHermitcrab)

박경태 : [GitHub](https://github.com/ParkKyungTae)

지호 : [GitHub](https://github.com/jiho4399)

## 🔍참고사항

### git
1. `git clone <본인 레포에서!! Fork한 Repo주소> <폴더명>`
2. `git remote -v` 하여 origin <주소> 2줄이 출력되는지 확인
3. `git remote add upstream <원본 Repo 주소>`
4. `git remote -v` 하여 origin <주소> 2줄, upstream <주소> 2줄이 출력되는지 확인
5. 터미널에 생성했던 폴더로 이동 (위에서 <폴더명> 에 해당하는 파일)
6. 본인의 이름 폴더로 이동
7. `git checkout -b <브랜치명>`
8. `git fetch upstream` 으로 Upstream Repo에 변경이 있었는지 확인
9. 있었다면 `git pull` 을 실행시켜 똑같이 반영
— 여기서 pull 뒤에 더 입력해야하라는 메세지 뜨면 각자 적용
10. `git add <파일명>`,
    `git commit -m '<커밋내용>'`, 
    `git push —set-upstream origin <자신의 브랜치명>` 
    순서로 입력하여 문제풀이 파일을 업로드
11. 깃허브로 돌아가서 Compare & pull request 를 눌러 원본 Repo에 반영을 요청
12. Pull Request 작성
13. 양식 바로 위에 Able to merge 가 쓰여있으면 Create Pull Request 누르기

(`git checkout <브랜치명>` 으로 현재 브랜치를 변경해서 사용 가능)


### 환경설정 서버에 올리기
```
pip freeze > requirements.txt  # freeze내용을 requirements.txt라는 파일로 내보내겠다.
pip install -r requirements.txt # pip로 install하는 걸 requirements.txt 를 읽어서 하겠다.
```
```
# 만약 conda를 쓰면
conda env export > filename.yaml # filename으로 env를 내보내겠다.
conda env create -f filename.yaml # filename으로 환경을 새로 만들겠다.

conda env update --prefix ./env --file filename.yaml --prune # update - 공식문서 참조
# https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
```