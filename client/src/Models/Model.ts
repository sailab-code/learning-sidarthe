export interface IModel {
    a: number;
    m: number;
    bound_loss_type: string;
    bound_reg: number;
    der_1st_reg: number;
    initial_values: Map<string, number[]>;
    integrator: string;
    learning_rates: Map<string, number>;
    loss_type: string;
    model_cls: string;
    model_name: string;
    momentum: boolean;
    region: string;
    started: string;
    t_inc: number;
    target_weights: Map<string, number>;
    train_size: number;
    val_len: number;
}

export type TInferenceIDs = "s" | "i" | "d" | "a" | "r" | "t" | "h_detected" | "e" | "r0"
export const inferenceIDs: TInferenceIDs[] = ["s", "i", "d", "a", "r", "t", "h_detected", "e", "r0"]