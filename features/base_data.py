import luigi
import luigi.target
import sqlite3
import pandas


class SplitData(luigi.Task):
    offset = luigi.IntParameter()
    rand_round = luigi.IntParameter()

    def output(self):
        ix = self.offset
        return [luigi.file.LocalTarget(path='/tmp/split_data/{}/train/{}.csv'.format(self.rand_round, ix)),
                luigi.file.LocalTarget(path='/tmp/split_data/{}/test/{}.csv'.format(self.rand_round, ix)),
                luigi.file.LocalTarget(path='/tmp/split_data/{}/score/{}.csv'.format(self.rand_round, ix))]

    def run(self):
        con = sqlite3.connect("/tmp/data.sqlite3")
        data = pandas.read_sql('''SELECT id,
                                         week_num,
                                         sales_depo,
                                         sales_channel,
                                         route_id,
                                         client_id,
                                         product_id,
                                         adjusted_demand,
                                         rand
                                    FROM data
                                   WHERE rand = ?
                                      ''', con=con, params=[self.offset])
        con.close()
        train_out, test_out, score_out = self.output()
        with train_out.open('w') as f:
            data[data.adjusted_demand.notnull() & (data.week_num < 8)].to_csv(f, index=False)
        with test_out.open('w') as f:
            data[data.adjusted_demand.notnull() & (data.week_num >= 8)].to_csv(f, index=False)
        with score_out.open('w') as f:
            data[data.adjusted_demand.isnull()].to_csv(f, index=False)


class BuildData(luigi.Task):
    rand_round = luigi.IntParameter()

    def requires(self):
        for i in range(100):
            yield SplitData(offset=i, rand_round=self.rand_round)

    def complete(self):
        for r in self.requires():
            if not r.complete():
                return False
        return True
