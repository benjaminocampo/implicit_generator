from transformers import pipeline
from itertools import cycle

from tempfile import TemporaryDirectory
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from sentence_transformers import SentenceTransformer
from src.preprocess import flatten_dict
from src.generate import generate_imp_hs

import hydra
import logging
import pandas as pd
import mlflow

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_experiment(cfg: DictConfig, run: mlflow.ActiveRun):
    """
    Conduct an experiment that generates hate speech examples for protected groups using
    a language model pipeline, and logs the results to MLflow.

    Parameters:
    - cfg (DictConfig): Configuration object that includes settings and parameters.
    - run (mlflow.ActiveRun): Active MLflow run to log artifacts.

    The function assumes that the configuration object `cfg` has the following structure:
    - input:
        - pretrained_path_or_model: Path or identifier for the pretrained model.
        - train_file: Path to the training data CSV file.
        - protected_groups: List of protected groups to generate hate speech examples for.
        - nof_shots: Number of examples to use as a prompt.
        - model_endpoint: Endpoint for the generation model.
        - secret_key: Secret key for accessing the model endpoint.
        - min_length_sent: Minimum length of generated sentences.
        - nof_gens: Number of generations to perform.
    """

    # Create a temporary directory to save output files
    with TemporaryDirectory() as tmpfile:
        output_dir = Path(tmpfile)

        # Initialize the classifier pipeline with the specified model
        classifier = pipeline("text-classification",
                              model=cfg.input.pretrained_path_or_model)

        lexicon = pd.read_csv(cfg.input.lexicon_file)

        sentence_sim = SentenceTransformer(cfg.input.sentence_sim)

        model_endpoint = f"https://api.openai.com/v1/engines/{cfg.input.engine}/completions"

        # Load and preprocess the training data
        data = pd.read_csv(cfg.input.train_file)
        data = data[data["implicit_layer"] == "Implicit HS"]

        nof_gens = 0
        adv_gens = []
        targets = []
        prompts = []

        # Cycle through the protected groups and generate hate speech examples
        for group in cycle(cfg.input.protected_groups):
            if nof_gens >= cfg.input.nof_gens:
                break

            data_with_target = data[data["sanitized_target"] == group]
            data_shots = data_with_target["text"].sample(cfg.input.nof_shots)

            examples = "\\n- ".join(data_shots)

            # Use the prompt template from the config, replacing placeholders with actual values
            # The last "-" is now part of the prompt template in the configuration file
            prompt = cfg.input.prompt_template.format(group=group.lower(), examples=examples)

            response = generate_imp_hs(prompt=prompt,
                                       shots=data_shots,
                                       secret_key=cfg.input.secret_key,
                                       model_endpoint=model_endpoint,
                                       classifier=classifier,
                                       sentence_sim=sentence_sim,
                                       lexicon=lexicon,
                                       end_token=cfg.input.end_token,
                                       weights=cfg.input.weights,
                                       num_beams=cfg.input.num_beams,
                                       max_length=cfg.input.max_length,
                                       length_penalty=cfg.input.length_penalty)

            for r in response:
                r = r[0][(len(prompt) + 1):].strip().replace("\\n", "")
                print(f"response: {r}\n")
                print(f"gen_id: {nof_gens}\n")
                adv_gens.append(r)
                targets.append(group)
                prompts.append(prompt)

                if len(r.split()) >= cfg.input.min_length_sent:
                    nof_gens += 1

        # Save generated data to a CSV file
        df_gens = pd.DataFrame({
            "prompt": prompts,
            "text": adv_gens,
            "target": targets,
        })
        output_csv_path = output_dir / f"gpt3-{cfg.input.engine}__shots={cfg.input.nof_shots}.csv"
        df_gens.to_csv(output_csv_path, index=False)

        # Log the output directory as an artifact
        mlflow.log_artifacts(str(output_dir), artifact_path="generated_data")


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg: DictConfig):
    """
    Main function to start the MLflow experiment with configurations managed by Hydra.
    
    Parameters:
    - cfg (DictConfig): A configuration object that includes all the parameters and settings
                        required for the experiment, loaded and managed by Hydra.
    """
    # Register a new resolver that can evaluate python expressions
    OmegaConf.register_new_resolver('eval', lambda x: eval(x))

    # Set up MLflow tracking URI if provided in the configuration
    if cfg.input.uri_path is not None:
        mlflow.set_tracking_uri(cfg.input.uri_path)
        # Ensure the tracking URI is correctly set
        assert cfg.input.uri_path == mlflow.get_tracking_uri(
        ), "Mismatch in MLflow tracking URI."

    # Log the current tracking URI
    logger.info(f"Current tracking uri: {cfg.input.uri_path}")

    # Set up the MLflow experiment using the name provided in the configuration
    mlflow.set_experiment(cfg.input.experiment_name)
    # Add a description tag to the experiment
    mlflow.set_experiment_tag('mlflow.note.content',
                              cfg.input.experiment_description)

    # Start an MLflow run with the specified run name
    with mlflow.start_run(run_name=cfg.input.run_name) as run:
        logger.info("Logging configuration as artifact")
        # Save the configuration to a temporary file and log it as an artifact
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            with open(config_path, "wt") as fh:
                print(OmegaConf.to_yaml(cfg, resolve=False), file=fh)
            mlflow.log_artifact(config_path)

        logger.info("Logging configuration parameters")
        # Flatten the nested configuration dictionary and log it to MLflow
        # This is necessary because MLflow expects a flat dictionary of parameters
        mlflow.log_params(
            flatten_dict(OmegaConf.to_container(cfg, resolve=False)))

        # Call the function to run the experiment with the current MLflow run context
        run_experiment(cfg, run)


if __name__ == '__main__':
    main()
