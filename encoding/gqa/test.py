from clingo.control import Control 
import json 
from encode import encode_question, sanitize
from itertools import islice

with open('../../data/gqa/questions/train_sampled_questions_2000.json') as f:
    questions = json.load(f)

with open('./theory.lp') as tf:
    theory = tf.read()

num_questions = len(questions)
correct = 0
incorrect = 0

def answer_is_correct(answer, correct_answer):
    if answer == sanitize(correct_answer): 
        return True
    elif (answer == 'to_the_right_of' and correct_answer == 'right') or \
        (answer == 'to_the_left_of' and correct_answer == 'left'):
        return True
    else:
        return False

for qid, question in islice(questions.items(), 0, num_questions):
    print(f"Solving question {qid}:")
    ctl = Control()
    ctl.add(theory)

    scene_encoding, question_encoding = encode_question(question)
    ctl.add(scene_encoding)
    ctl.add(question_encoding)

    answer = [None]
    def on_model(model):
        answer[0] = model.symbols(shown=True)[0].arguments[0].name

    ctl.ground()
    result = ctl.solve(on_model=on_model)

    if result.satisfiable:
        if(answer_is_correct(answer[0], question['answer'])):
            print(f"Correct answer: {answer[0]}")
            correct = correct + 1
        else: 
            print(f"Incorrect answer: {answer[0]} (correct: {question['answer']})")
            incorrect = incorrect + 1
    else: 
        incorrect = incorrect + 1
        print("UNSAT")

print("===============================")
print(f"Total questions: {num_questions}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Percentage: {correct/num_questions*100}%")