def get_project_loss_and_eval_dict() -> dict:
    """ Returns two dictionaries. The first dictionary contains custom
        loss functions for this project and associated names to reference
        them (keys). The second dictionary contains custom metric 
        functions for this project and associated names to references
        them.
    """
    loss_dict = {}
    eval_dict = {}
    return loss_dict, eval_dict