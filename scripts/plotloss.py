import argparse
import sys
import matplotlib.pyplot as plt
import re

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_file",
        help = "path to log file"
    )

    args = parser.parse_args()

    f = open(args.log_file)
    lines = [line.rstrip("\n") for line in f.readlines()]

    iters = []
    loss = []
    accuracy = []
    for line in lines:
        if len(line) == 0:
            continue
        #parts = line.split("\t")
        parts = re.split("\t | ", line)
        for i in range(0,len(parts)):
            if parts[i]=="Accuracy:":
                accuracy.append(float(parts[i+1])*100)
            if parts[i]=="Loss:":
                loss.append(float(parts[i+1]))
            if parts[i]=="Iteration:":
                iters.append(float(parts[i+1]))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(iters, loss, 'g-')
    ax2.plot(iters, accuracy, 'b-')
    ax1.set_xlabel('iters')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

