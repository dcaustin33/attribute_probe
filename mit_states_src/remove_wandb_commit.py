import os

if __name__ == '__main__':
    #find all folders with the name wandb and delete them
    for root, dirs, files in os.walk(".", topdown=False):
        for name in dirs:
            if name == "wandb":
                print(os.path.join(root, name))
                #remove the directory
                os.system("rm -rf " + os.path.join(root, name))
                #os.rmdir(os.path.join(root, name))

    try:
        os.system("git add -- . ':!*.pt' ':!*.tar' ':!*.bin' ':!*.jpeg' ':!*.webp'")
        inp = input("Commit message: ")
        os.system("git commit -m '" + inp + "'")
        os.system("git push")
    except:
        print("Error")