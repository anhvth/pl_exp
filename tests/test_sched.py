from ple import *

# fn = lr_function_by_epoch(
#     4, 1000, 2, 1, 
# )


def plot_lr_step_schedule(fm, lr=1e-5, total_step=1000):
    import matplotlib.pyplot as plt
    lrs = []
    steps = total_step
    for step in range(steps):
        lrs.append(fn(step)*lr)
    # print(f'{min(lrs)=:0.5f}, {max(lrs)=:0.5f}')
    plt.plot(range(steps), lrs)
    # plt.show()
    plt.savefig('/tmp/lr.png')
    print('/tmp/lr.png')
fn = _lr_function_by_step(
    1000, 2
)
plot_lr_step_schedule(fn)