"""
Generate evaluation data
"""

if __name__ == '__main__':
    from scripts.evals import eval_utils


    ident = 'paper'
    props = [
        'YoungsModulus293K_Density293K', 
        'YoungsModulus293K_Density293K_RefractiveIndex', 
        'LogViscosity_Tg', 
        'LogViscosity_Tg_t'
        ]
    exps = ['only_p1_single', 'aux_p_pad', 'aux_p_random_pad']

    Samples = eval_utils.Samples(
        ident, 
        props, 
        exps, 
        seed = 42    # Alternatively 1 or 100; Does not change the results.
        )
    Samples.samples_all()
    Samples.stats()

    Errors = eval_utils.Errors(ident, props, exps, seeds = [1, 42, 100])
    Errors.save_errors()
    Errors.linear_fit()
    Errors.plot_YM()
    Errors.plot_YM_MAE()
    Errors.plot_LogViscosity()