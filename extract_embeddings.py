from entity_embeddings import Config, Embedder, TargetType


def main(data_path='ready_to_train_50.csv'):

    config = Config.make_default_config(csv_path=data_path,
                                    target_name='paid_loss_trended',
                                    target_type=TargetType.BINARY_CLASSIFICATION,
                                    train_ratio=0.9, 
                                    epochs=100)

    embedder = Embedder(config)
    embedder.perform_embedding()
    #import pdb; pdb.set_trace()


if __name__ == '__main__':
	main()